"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        u_pooler=None,
        decoder=None,
        model_mode="lrc_jepa",
        u_mode="clip_avg",
        diff_z=False,
        eval_encode_u=False,
        channels_last=True,
    ):
        super().__init__()

        if model_mode != "lrc_jepa":
            raise ValueError(f"Unknown model_mode: {model_mode}")

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.u_pooler = u_pooler
        self.decoder = decoder
        self.model_mode = model_mode
        self.use_u = model_mode == "lrc_jepa"
        self.u_mode = u_mode
        self.use_diff_z = diff_z
        self.eval_encode_u = eval_encode_u
        self.channels_last = channels_last
        if self.channels_last:
            self.to(memory_format=torch.channels_last)

    def encode(self, info, return_u=None):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        if self.channels_last and pixels.ndim == 4:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        tokens = output.last_hidden_state
        cls_token = tokens[:, 0]  # cls token
        z = self.projector(cls_token)
        z = rearrange(z, "(b t) d -> b t d", b=b)

        info["z"] = z
        info["emb"] = z

        if return_u is None:
            return_u = self.use_u
        if return_u and self.use_u and self.u_pooler is not None:
            u = self.u_pooler(tokens)
            info["u"] = rearrange(u, "(b t) q d -> b t q d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(
                info["action"], info.get("dataset_id")
            )

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding; `diff_z` uses a residual z update.
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return emb + preds if self.use_diff_z else preds

    def reconstruct(self, z, u=None, u_mode=None):
        """Reconstruct patchified pixels from z and u tokens."""
        if self.decoder is None:
            return None

        if self.use_u:
            if u is None:
                raise ValueError("LRC-JEPA reconstruction requires u tokens")
            mode = u_mode or self.u_mode
            if mode == "clip_avg":
                u = u.mean(dim=1)
            elif mode == "per_frame":
                pass
            else:
                raise ValueError(f"Unknown u_mode: {mode}")
        else:
            u = None

        return self.decoder(z, u)

    @staticmethod
    def patchify_pixels(pixels, patch_size):
        """Convert images into flattened pixel patches."""
        h, w = pixels.shape[-2:]
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"Image size {(h, w)} must be divisible by patch_size={patch_size}"
            )
        return rearrange(
            pixels,
            "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )


    def reconstruction_loss(self, pixels, z, u=None, patch_size=None, u_mode=None):
        """Pixel-patch reconstruction loss over all patches."""
        pred_pixels = self.reconstruct(z, u, u_mode=u_mode)
        target_pixels = self.patchify_pixels(pixels.float(), patch_size)
        return F.mse_loss(pred_pixels, target_pixels)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3, return_u=None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        B, S, T = action_sequence.shape[:3]

        init_emb = info.get("init_emb")
        if init_emb is None:
            pixels = info["pixels"]
            _init = {"pixels": pixels[:, 0] if pixels.ndim == 6 else pixels}
            if "dataset_id" in info:
                dataset_id = info["dataset_id"]
                _init["dataset_id"] = dataset_id[:, 0] if dataset_id.ndim > 1 else dataset_id
            _init = self.encode(_init, return_u=return_u)
            init_emb = _init["emb"]
            info["init_emb"] = init_emb

        H = init_emb.size(1)
        n_steps = T - H
        if n_steps < 0:
            raise ValueError(
                f"action_sequence length {T} must be at least history length {H}"
            )

        emb = init_emb.unsqueeze(1).expand(B, S, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(action_sequence, "b s ... -> (b s) ...")
        dataset_id = info.get("dataset_id")
        if torch.is_tensor(dataset_id):
            if dataset_id.ndim > 1:
                dataset_id = rearrange(dataset_id, "b s ... -> (b s) ...")
            else:
                dataset_id = dataset_id[:, None].expand(B, S).reshape(B * S)
        act_emb = self.action_encoder(act, dataset_id)

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for step in range(n_steps + 1):
            prefix_end = H + step
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, max(0, prefix_end - HS) : prefix_end]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        pred_final = pred_emb[..., -1, :]
        if goal_emb.ndim == pred_emb.ndim - 1:
            goal_final = goal_emb[:, None, -1, :]
        else:
            goal_final = goal_emb[..., -1, :]
        return (pred_final - goal_final.detach()).pow(2).sum(dim=-1)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if "goal_emb" not in info_dict:
            goal_pixels = info_dict["goal"]
            goal = {"pixels": goal_pixels[:, 0] if goal_pixels.ndim == 6 else goal_pixels}
            goal = self.encode(goal, return_u=self.eval_encode_u)
            info_dict["goal_emb"] = goal["emb"]

        if "init_emb" not in info_dict:
            init_pixels = info_dict["pixels"]
            init = {"pixels": init_pixels[:, 0] if init_pixels.ndim == 6 else init_pixels}
            if "dataset_id" in info_dict:
                dataset_id = info_dict["dataset_id"]
                init["dataset_id"] = dataset_id[:, 0] if dataset_id.ndim > 1 else dataset_id
            init = self.encode(init, return_u=self.eval_encode_u)
            info_dict["init_emb"] = init["emb"]

        info_dict = self.rollout(
            info_dict,
            action_candidates,
            return_u=self.eval_encode_u,
        )

        cost = self.criterion(info_dict)
        
        return cost
