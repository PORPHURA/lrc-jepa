from runtime import configure_runtime
configure_runtime()

import os
from functools import partial
from pathlib import Path
import hydra
import lightning as pl
from hydra.core.hydra_config import HydraConfig

import stable_pretraining as spt
import torch
from omegaconf import OmegaConf, open_dict

from compat import patch_stable_pretraining_sidecar_atomic_write
from module import SIGReg, VICRegLoss
from utils import (
    LRCModelCheckpoint,
    prepare_resume_checkpoint,
    prepare_training_dataset,
    SaveCkptCallback,
)

patch_stable_pretraining_sidecar_atomic_write()


def dataloader_kwargs(loader_cfg):
    kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
    if kwargs["num_workers"] == 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = None
    return kwargs


def optimizer_kwargs(optimizer_cfg, lr=None):
    kwargs = OmegaConf.to_container(optimizer_cfg, resolve=True)
    kwargs.pop("decoder_lr", None)
    if lr is not None:
        kwargs["lr"] = lr
    return kwargs


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    sigreg_weight = cfg.loss.sigreg.weight
    u_weight = cfg.loss.u.weight
    rec_weight = cfg.loss.rec.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    z = output["z"]
    act_emb = output["act_emb"]

    ctx_emb = z[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = z[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LRC-JEPA loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(z.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]

    if "u" in output and u_weight > 0:
        output["u_vicreg_loss"] = self.u_vicreg(output["u"])
        output["loss"] = output["loss"] + u_weight * output["u_vicreg_loss"]

    if self.model.decoder is not None and rec_weight > 0:
        rec_z = z.detach() if cfg.loss.rec.detach_latents else z
        rec_u = output.get("u")
        output["rec_loss"] = self.model.reconstruction_loss(
            pixels=batch["pixels"],
            z=rec_z,
            u=rec_u,
            patch_size=cfg.loss.rec.patch_size,
            u_mode=cfg.u_mode,
        )
        output["loss"] = output["loss"] + rec_weight * output["rec_loss"]

    losses = {f"{stage}/{key}": value.detach() for key, value in output.items() if "loss" in key}
    self.log_dict(losses, on_step=True, sync_dist=True, logger=False, prog_bar=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="pusht")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.performance.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.performance.cudnn_benchmark

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spt.set(
        default_loggers={"registry": False},
        default_callbacks={name: False for name in (
            "registry", "logging", "env_dump", "trainer_info", "sklearn_checkpoint",
            "wandb_checkpoint", "trackio_checkpoint", "swanlab_checkpoint",
            "module_summary", "slurm_info", "hf_checkpoint",
        )},
    )
    # The setter accepts None; spt.set(cache_dir=None) leaves the default unchanged.
    spt.get_config().cache_dir = None

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset, action_dims = prepare_training_dataset(
        dataset_cfg,
        img_size=cfg.img_size,
        cache_dir=cache_dir,
        resize_images=cfg.preprocess.resize_images,
    )
    
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * int(action_dims[0])
        if cfg.loader.num_workers == 0:
            cfg.loader.persistent_workers = False
            cfg.loader.prefetch_factor = None
        if cfg.val_loader.num_workers == 0:
            cfg.val_loader.persistent_workers = False
            cfg.val_loader.prefetch_factor = None

    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader_kwargs = dataloader_kwargs(cfg.loader)
    val_loader_kwargs = dataloader_kwargs(OmegaConf.merge(cfg.loader, cfg.val_loader))

    train = torch.utils.data.DataLoader(
        train_set,
        **train_loader_kwargs,
        shuffle=False,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **val_loader_kwargs,
        shuffle=False,
        drop_last=True,
    )
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    scheduler_max_steps = max(2, int(cfg.scheduler.max_steps))
    scheduler_warmup_steps = max(
        1,
        min(int(cfg.scheduler.warmup_steps), scheduler_max_steps - 1),
    )
    lr_scheduler = {
        "type": cfg.scheduler.type,
        "warmup_steps": scheduler_warmup_steps,
        "max_steps": scheduler_max_steps,
        "warmup_start_lr": cfg.scheduler.warmup_start_lr,
        "eta_min": cfg.scheduler.eta_min,
    }

    optimized_modules = [
        "encoder",
        "predictor",
        "action_encoder",
        "projector",
        "pred_proj",
        "u_pooler",
    ]
    train_decoder = (
        cfg.model.decoder is not None
        and cfg.loss.rec.weight > 0
    )
    optimizers = {
        "model_opt": {
            "modules": rf"model\.({'|'.join(optimized_modules)})($|\.)",
            "optimizer": optimizer_kwargs(cfg.optimizer),
            "scheduler": dict(lr_scheduler),
            "interval": "epoch",
        },
    }
    if train_decoder:
        optimizers["decoder_opt"] = {
            "modules": r"model\.decoder($|\.)",
            "optimizer": optimizer_kwargs(cfg.optimizer, lr=cfg.optimizer.get("decoder_lr", cfg.optimizer.lr)),
            "scheduler": dict(lr_scheduler),
            "interval": "epoch",
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        u_vicreg=VICRegLoss(**cfg.loss.u.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    with open_dict(cfg):
        cfg.trainer.default_root_dir = str(run_dir)

    object_dump_callback = SaveCkptCallback(
        save_dir=run_dir,
        epoch_interval=1,
    )
    checkpoint_callback = LRCModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="{epoch}-{step}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback, object_dump_callback],
        logger=False,
        enable_checkpointing=True,
    )

    ckpt_path = prepare_resume_checkpoint(run_dir / "checkpoints" / "last.ckpt")
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
        seed=cfg.seed,
    )

    manager()
    return


if __name__ == "__main__":
    run()
