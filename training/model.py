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
LayerNorm = partial(nn.LayerNorm, epsilon=1e-5)


class CasualAttention(nn.Module):
    dropout_rate: float = 0.2
    num_heads: int = 8
    use_bias: bool = True
    proj_kernel_init_norm: float = 1.0
    reduce_ops_dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = None

    @nn.compact
    def __call__(self, x, *, train=True):
        batch, seq_length, embd_dim = x.shape

        q = HiddenDense(embd_dim, use_bias=self.use_bias, dtype=self.params_dtype)(x)
        k = HiddenDense(embd_dim, use_bias=self.use_bias, dtype=self.params_dtype)(x)
        v = HiddenDense(embd_dim, use_bias=self.use_bias, dtype=self.params_dtype)(x)

        q = q.reshape(
            batch, seq_length, self.num_heads, embd_dim // self.num_heads
        ).transpose([0, 2, 1, 3])
        k = k.reshape(
            batch, seq_length, self.num_heads, embd_dim // self.num_heads
        ).transpose([0, 2, 1, 3])
        v = v.reshape(
            batch, seq_length, self.num_heads, embd_dim // self.num_heads
        ).transpose([0, 2, 1, 3])

        # batch, num_head, seq_length, seq_length
        dot_product = q @ k.transpose([0, 1, 3, 2]) * (1.0 / jnp.sqrt(k.shape[-1]))

        # batch, num_head, seq_length, seq_length
        casual_mask = nn.make_causal_mask(
            x=jnp.ones((batch, self.num_heads, seq_length))
        ).squeeze()

        masked_dot_product = jnp.where(
            casual_mask, dot_product, jnp.finfo(dot_product.dtype).min
        )

        # force masked_dot_product dtype to full precision when running on GPU
        attn_scores = jax.nn.softmax(masked_dot_product.astype(self.reduce_ops_dtype))
        attn_scores = attn_scores.astype(masked_dot_product.dtype)

        attn_scores = nn.Dropout(self.dropout_rate)(
            attn_scores, deterministic=not train
        )

        # batch, num_head, seq_length, embd_dim // num_heads
        attn_embeddings = attn_scores @ v

        # batch, seq_length, embd_dim
        attn_embeddings = attn_embeddings.transpose([0, 2, 1, 3]).reshape(
            (batch, seq_length, -1)
        )

        # batch, seq_length, embd_dim

        proj_dense = make_dense(kernel_init_std=0.02 / self.proj_kernel_init_norm)
        post_projection_attn_embeddings = proj_dense(
            embd_dim, use_bias=self.use_bias, dtype=self.params_dtype
        )(attn_embeddings)

        # batch, seq_length, embd_dim
        attn = nn.Dropout(self.dropout_rate)(
            post_projection_attn_embeddings, deterministic=not train
        )

        return attn


class MLP(nn.Module):
    input_factor: int = 4
    dropout_rate: float = 0.2
    use_bias: bool = True
    proj_kernel_init_norm: int = 1
    params_dtype: jnp.dtype = None

    @nn.compact
    def __call__(self, x, train=True):
        embd_dim = x.shape[-1]
        x = HiddenDense(
            self.input_factor * embd_dim,
            use_bias=self.use_bias,
            dtype=self.params_dtype,
        )(x)

        x = nn.activation.gelu(x)

        x = make_dense(kernel_init_std=0.02 / self.proj_kernel_init_norm)(
            embd_dim, use_bias=self.use_bias, dtype=self.params_dtype
        )(x)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        return x


class AttentionBlock(nn.Module):
    use_bias: bool = True
    proj_kernel_init_norm: float = 1.0
    reduce_ops_dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = None

    @nn.compact
    def __call__(self, x, train=True):
        x_m = LayerNorm(
            use_bias=self.use_bias, use_scale=True, dtype=self.params_dtype
        )(x.astype(self.reduce_ops_dtype))
        x_m = x_m.astype(x.dtype)

        x = (
            CasualAttention(
                use_bias=self.use_bias,
                proj_kernel_init_norm=self.proj_kernel_init_norm,
                reduce_ops_dtype=self.reduce_ops_dtype,
                params_dtype=self.params_dtype,
            )(x_m, train=train)
            + x
        )

        x_m = LayerNorm(
            use_bias=self.use_bias, use_scale=True, dtype=self.params_dtype
        )(x.astype(self.reduce_ops_dtype))
        x_m = x_m.astype(x.dtype)

        x = (
            MLP(
                use_bias=self.use_bias,
                proj_kernel_init_norm=self.proj_kernel_init_norm,
                params_dtype=self.params_dtype,
            )(x_m, train=train)
            + x
        )

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_layers: int = 12
    num_heads: int = 12
    embd_dim: int = 768
    dropout_rate: float = 0.0
    use_bias: bool = True  # True: bias in Dense and LayerNorms, like GPT-2. False: a bit better and faster
    reduce_ops_dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = None


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.token_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embd_dim,
            embedding_init=normal_initializer(std=0.02),
            dtype=self.config.params_dtype,
        )
        self.positional_embeddings = nn.Embed(
            num_embeddings=self.config.block_size,
            features=self.config.embd_dim,
            embedding_init=normal_initializer(std=0.02),
            dtype=self.config.params_dtype,
        )

        self.dropout = nn.Dropout(self.config.dropout_rate)

        self.blocks = [
            AttentionBlock(
                use_bias=self.config.use_bias,
                proj_kernel_init_norm=jnp.sqrt(2 * self.config.num_layers),
                reduce_ops_dtype=self.config.reduce_ops_dtype,
                params_dtype=self.config.params_dtype,
            )
            for _ in range(self.config.num_layers)
        ]

        self.layer_norm = LayerNorm(
            use_bias=self.config.use_bias,
            use_scale=True,
            dtype=self.config.params_dtype,
        )

    def __call__(self, x, *, train=True, top_k=None):
        batch, seq_length = x.shape
        assert seq_length <= self.config.block_size

        positions = self.positional_embeddings(jnp.arange(0, stop=seq_length, step=1))
        embeddings = self.token_embeddings(x) + positions
        embeddings = self.dropout(embeddings, deterministic=not train)
        for block in self.blocks:
            embeddings = block(embeddings, train=train)

        if top_k:
            embeddings = embeddings[:, -top_k:, :]

        embeddings = self.layer_norm(embeddings.astype(self.config.reduce_ops_dtype))
        embeddings = embeddings.astype(positions.dtype)

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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    config = GPTConfig(
        block_size=32,
        num_layers=8,
        num_heads=4,
        embd_dim=128,
        dropout_rate=0.0,
        use_bias=False,
    )

    random_input = jax.random.randint(key, (2, 8), minval=0, maxval=config.vocab_size)

    model = GPT(config)
    state = model.init(key, x=random_input, train=False)
    output = model.apply(state, x=random_input, train=False)

    assert output.shape == random_input.shape + (config.vocab_size,)

    tokens = jax.jit(model.generate, static_argnums=(3,))(
        key, state["params"], context=random_input, max_new_tokens=10
    )
