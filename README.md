# flash-nanoGPT (Under development)

a jax (flax) re-write of Andrej Karpathy NanoGPT, this repository will hold a collection of Jax/Flax new features like :
Pallas kernel language for flashAttention on TPU, Data and tensor sharding with Jax on TPU

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
- [ ] Loading GPTs pre-trained models
- [ ] Multi-host training
- [ ] Reproducing results on OpenWebText dataset
- [ ] Fine tuning GPT-2 weights on Shakespear dataset
- [ ] Profiling/Benchmarking and speeding up iteration
- [ ] Flash attention with Pallas