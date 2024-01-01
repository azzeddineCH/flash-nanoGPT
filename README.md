# ⚡⚡ flash-nanoGPT ⚡⚡

a jax/flax re-write of [Andrej Karpathy NanoGPT](https://github.com/karpathy/nanoGPT), this repository will hold a collection of Jax/Flax new features including :
[Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) kernel language for flashAttention on TPU, sharding using the new [shmap](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) primitive.

A special thanks to the [TPU Research Cloud team](https://sites.research.google/trc/about/) for providing serval TPU v2-8 and v3-32 instances!

## Todos

- [x] GPT model in flax
- [x] Core data pipeline
- [x] Core training loop
- [x] Mixed precision training
- [x] Gradient accumulation
- [x] Data sharding across GPUs/TPUs
- [x] Loading and Saving checkpoints
- [x] Reproduce the results on shakespear-char dataset
- [x] TF Record reader/write with support for GCP bucket
- [x] Multi-host training
- [ ] Reproducing results on OpenWebText dataset
- [ ] Loading GPTs pre-trained models
- [ ] Fine tuning GPT-2 weights on Shakespear dataset
- [ ] inference script 
- [ ] Profiling/Benchmarking and speeding up iteration/inference
- [ ] Providing Google cloud manuall
- [ ] Flash attention with Pallas
