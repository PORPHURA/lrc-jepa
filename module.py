import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class SDPAMultiheadAttention(nn.Module):
    """Batch-first multi-head attention backed directly by PyTorch SDPA.

    Parameter names mirror nn.MultiheadAttention so existing checkpoints load.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        if not batch_first:
            raise ValueError("SDPAMultiheadAttention only supports batch_first=True")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        need_weights=False,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
    ):
        if attn_mask is not None or key_padding_mask is not None:
            raise NotImplementedError("Masks are not used by LRC-JEPA SDPA attention")

        q_w, k_w, v_w = self.in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, q_w, q_b)
        k = F.linear(key, k_w, k_b)
        v = F.linear(value, v_w, v_b)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=is_causal
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out, None


class CrossAttentionBlock(nn.Module):
    """Cross-attention block used by the reconstruction decoder."""

    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = SDPAMultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, context):
        context = self.context_norm(context)
        attn, _ = self.cross_attn(
            self.query_norm(x),
            context,
            context,
            need_weights=False,
        )
        x = x + attn
        x = x + self.mlp(x)
        return x


class AttentivePool(nn.Module):
    """Pool ViT tokens into one or more context tokens with learned queries."""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_queries=4,
        heads=4,
        mlp_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.input_proj = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, output_dim) * 0.02)
        self.token_norm = nn.LayerNorm(output_dim)
        self.query_norm = nn.LayerNorm(output_dim)
        self.attn = SDPAMultiheadAttention(
            output_dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp = FeedForward(output_dim, mlp_dim or 4 * output_dim, dropout=dropout)

    def forward(self, tokens):
        """
        tokens: (B, N, D)
        returns: (B, Q, output_dim)
        """
        tokens = self.input_proj(tokens)
        queries = self.query_tokens.expand(tokens.size(0), -1, -1)
        tokens = self.token_norm(tokens)
        pooled, _ = self.attn(
            self.query_norm(queries),
            tokens,
            tokens,
            need_weights=False,
        )
        pooled = pooled + self.mlp(pooled)
        return pooled


class VICRegLoss(nn.Module):
    """VICReg-style stability and anti-collapse loss for context tokens."""

    def __init__(
        self,
        invariance_weight=25.0,
        variance_weight=25.0,
        covariance_weight=1.0,
        variance_target=1.0,
        eps=1e-4,
    ):
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target
        self.eps = eps

    def forward(self, u):
        """
        u: (B, T, Q, D) or (B, T, D)
        """
        if u.ndim == 4:
            u_flat = rearrange(u, "b t q d -> b t (q d)")
        elif u.ndim == 3:
            u_flat = u
        else:
            raise ValueError(f"Expected u with 3 or 4 dims, got shape {tuple(u.shape)}")

        clip_repr = u_flat.mean(dim=1)
        invariance = (u_flat - clip_repr[:, None]).pow(2).mean()

        if clip_repr.size(0) < 2:
            variance = clip_repr.new_zeros(())
            covariance = clip_repr.new_zeros(())
        else:
            std = torch.sqrt(clip_repr.var(dim=0, unbiased=False) + self.eps)
            variance = F.relu(self.variance_target - std).mean()

            centered = clip_repr - clip_repr.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / (clip_repr.size(0) - 1)
            covariance = off_diagonal(cov).pow(2).sum() / clip_repr.size(1)

        return (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )


def off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CrossAttentionDecoder(nn.Module):
    """CLSDecoder-style pixel decoder conditioned on z, optionally plus context u."""

    def __init__(
        self,
        latent_dim,
        patch_dim,
        num_patches,
        decoder_dim=None,
        depth=2,
        heads=4,
        mlp_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        decoder_dim = decoder_dim or latent_dim
        mlp_dim = mlp_dim or 4 * decoder_dim
        self.num_patches = num_patches
        self.queries = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.normal_(self.queries, std=0.02)

        self.context_proj = nn.Sequential(
            nn.Linear(latent_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": SDPAMultiheadAttention(
                            decoder_dim,
                            heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "query_norm": nn.LayerNorm(decoder_dim),
                        "context_norm": nn.LayerNorm(decoder_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(decoder_dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_dim, decoder_dim),
                            nn.Dropout(dropout),
                        ),
                        "mlp_norm": nn.LayerNorm(decoder_dim),
                    }
                )
            )

        self.to_patch = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, patch_dim),
        )

    def forward(self, z, u=None):
        """
        z: (B, T, D)
        u: optional (B, T, Q, D) or (B, Q, D)
        returns: (B, T, N, patch_dim)
        """
        b, t, _ = z.shape

        z_tokens = rearrange(z, "b t d -> (b t) 1 d")
        if u is None:
            context = z_tokens
        else:
            if u.ndim == 3:
                u = u[:, None].expand(-1, t, -1, -1)
            elif u.ndim != 4:
                raise ValueError(f"Expected u with 3 or 4 dims, got shape {tuple(u.shape)}")
            u_tokens = rearrange(u, "b t q d -> (b t) q d")
            context = torch.cat([z_tokens, u_tokens], dim=1)
        kv = self.context_proj(context)
        q = self.queries.expand(b * t, -1, -1)

        for layer in self.layers:
            kv_norm = layer["context_norm"](kv)
            attn_out = layer["cross_attn"](
                layer["query_norm"](q),
                kv_norm,
                kv_norm,
                need_weights=False,
            )[0]
            q = q + attn_out
            q = q + layer["mlp"](layer["mlp_norm"](q))

        patches = self.to_patch(q)
        return rearrange(patches, "(b t) n d -> b t n d", b=b, t=t)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x, dataset_id=None):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)

class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
