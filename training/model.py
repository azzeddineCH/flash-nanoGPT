from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
from jax import numpy as jnp


def normal_initializer(mean: float = 0, std: float = 1):
    def init(key, shape, dtype):
        return jax.random.normal(key, shape=shape, dtype=dtype) * std + mean

    return init


def make_dense(kernel_init_mean: float = 0, kernel_init_std: float = 1):
    return partial(
        nn.Dense,
        kernel_init=normal_initializer(mean=kernel_init_mean, std=kernel_init_std),
    )


HiddenDense = make_dense(kernel_init_std=0.02)
LayerNorm = partial(nn.LayerNorm, epsilon=1e-5, use_scale=True)


class CasualAttention(nn.Module):
    dropout_rate: float
    num_heads: int
    use_bias: bool
    proj_kernel_init_norm: float
    reduce_ops_dtype: jnp.dtype
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, *, train=True):
        batch, seq_length, embd_dim = x.shape

        head_dim = embd_dim // self.num_heads
        qkv = HiddenDense(
            embd_dim * 3,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            name="c_attn",
        )(x)
        qkv = qkv.reshape(batch, seq_length, 3 * self.num_heads, head_dim)
        q, k, v = jnp.split(qkv, indices_or_sections=3, axis=2)

        dot_product = jnp.einsum("...qhd,...khd->...hqk", q, k) * (
            1.0 / jnp.sqrt(head_dim)
        ).astype(x.dtype)

        casual_mask = nn.make_causal_mask(
            x=jnp.ones((batch, self.num_heads, seq_length))
        ).squeeze()
        masked_dot_product = jnp.where(
            casual_mask, dot_product, jnp.finfo(dot_product.dtype).min
        )

        attn_scores = jax.nn.softmax(masked_dot_product.astype(self.reduce_ops_dtype))
        attn_scores = attn_scores.astype(masked_dot_product.dtype)

        attn_scores = nn.Dropout(self.dropout_rate)(
            attn_scores, deterministic=not train
        )

        # batch, seq_length, embd_dim
        attn_embeddings = jnp.einsum("...hqk, ...khd->...qhd", attn_scores, v).reshape(
            (batch, seq_length, -1)
        )

        proj_dense = make_dense(kernel_init_std=0.02 / self.proj_kernel_init_norm)
        post_projection_attn_embeddings = proj_dense(
            embd_dim,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            name="c_proj",
        )(attn_embeddings)

        # batch, seq_length, embd_dim
        attn = nn.Dropout(self.dropout_rate)(
            post_projection_attn_embeddings, deterministic=not train
        )

        return attn


class MLP(nn.Module):
    input_factor: int
    dropout_rate: float
    use_bias: bool
    proj_kernel_init_norm: float
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, train=True):
        embd_dim = x.shape[-1]
        x = HiddenDense(
            self.input_factor * embd_dim,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            name="c_fc",
        )(x)

        x = nn.activation.gelu(x)

        x = make_dense(kernel_init_std=0.02 / self.proj_kernel_init_norm)(
            embd_dim,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            name="c_proj",
        )(x)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        return x


class AttentionBlock(nn.Module):
    num_heads: int
    use_bias: bool
    dropout_rate: float
    proj_kernel_init_norm: float
    reduce_ops_dtype: jnp.dtype
    param_dtype: jnp.dtype

    def setup(self):
        self.l1 = LayerNorm(
            use_bias=self.use_bias, param_dtype=self.param_dtype, name="ln_1"
        )
        self.attention = CasualAttention(
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            proj_kernel_init_norm=self.proj_kernel_init_norm,
            reduce_ops_dtype=self.reduce_ops_dtype,
            param_dtype=self.param_dtype,
            name="attn",
        )
        self.l2 = LayerNorm(
            use_bias=self.use_bias, param_dtype=self.param_dtype, name="ln_2"
        )
        self.mlp = MLP(
            input_factor=4,
            use_bias=self.use_bias,
            proj_kernel_init_norm=self.proj_kernel_init_norm,
            param_dtype=self.param_dtype,
            dropout_rate=self.dropout_rate,
        )

    def __call__(self, x, train=True):
        x_l1 = self.l1(x.astype(self.reduce_ops_dtype))
        x_l1 = x_l1.astype(x.dtype)

        x = self.attention(x_l1, train=train) + x

        x_l2 = self.l2(x.astype(self.reduce_ops_dtype))
        x_l2 = x_l2.astype(x.dtype)

        x = self.mlp(x_l2, train=train) + x

        return x


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_layers: int
    num_heads: int
    embd_dim: int
    dropout_rate: float
    use_bias: bool  # True: bias in Dense and LayerNorms, like GPT-2. False: a bit better and faster
    reduce_ops_dtype: jnp.dtype
    param_dtype: jnp.dtype


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.token_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embd_dim,
            embedding_init=normal_initializer(std=0.02),
            param_dtype=self.config.param_dtype,
            name="wte",
        )
        self.positional_embeddings = nn.Embed(
            num_embeddings=self.config.block_size,
            features=self.config.embd_dim,
            embedding_init=normal_initializer(std=0.02),
            param_dtype=self.config.param_dtype,
            name="wpe",
        )

        self.dropout = nn.Dropout(self.config.dropout_rate)

        self.blocks = [
            AttentionBlock(
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
                use_bias=self.config.use_bias,
                proj_kernel_init_norm=jnp.sqrt(2 * self.config.num_layers),
                reduce_ops_dtype=self.config.reduce_ops_dtype,
                param_dtype=self.config.param_dtype,
                name=f"h.{i}",
            )
            for i in range(self.config.num_layers)
        ]

        self.layer_norm = LayerNorm(
            use_bias=self.config.use_bias,
            param_dtype=self.config.param_dtype,
            name="ln_f",
        )

    def __call__(self, x, *, train=True, top_k=None):
        batch, seq_length = x.shape
        assert seq_length <= self.config.block_size

        positions_embeddings = self.positional_embeddings(
            jnp.arange(0, stop=seq_length, step=1)
        )
        embeddings = self.token_embeddings(x) + positions_embeddings
        embeddings = self.dropout(embeddings, deterministic=not train)
        for block in self.blocks:
            embeddings = block(embeddings, train=train)

        if top_k:
            embeddings = embeddings[:, -top_k:, :]

        embeddings = embeddings.astype(self.config.reduce_ops_dtype)
        embeddings = self.layer_norm(embeddings)
        embeddings = embeddings.astype(positions_embeddings.dtype)

        logits = self.token_embeddings.attend(embeddings)

        return logits

    def generate(self, rng_key, params, context, max_new_tokens, temperature=0.1):
        for i in range(max_new_tokens):
            token_key = jax.random.fold_in(rng_key, i)
            trunc_context = (
                context
                if context.shape[-1] <= self.config.block_size
                else context[:, -self.config.block_size :]
            )
            logits = self.apply(
                {"params": params}, x=trunc_context, train=False, top_k=1
            )[:, -1, :]
            next_token = jax.random.categorical(token_key, logits / temperature)
            context = jnp.concatenate([context, next_token[..., None]], axis=-1)

        return context
