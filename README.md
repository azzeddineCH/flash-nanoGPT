# flash-nanoGPT (Under developement)

a jax (flax) re-write of Andrej Karpathy NanoGPT, this repository will hold a collection of Jax/Flax new features like :
Pallas kernel language for flashAttention on TPU, Data and tensor sharding with Jax on TPU

## Todos
- [x] GPT model in flax
- [x] Core data pipeline
- [x] Core training loop
- [x] Mixed precision training
- [ ] Gradient accumulation
- [ ] Batch pre-fetching
- [ ] Data sharding across GPUs/TPUs
- [ ] Reproducing results on shakespear dataset
- [ ] Reproducing results on OpenWebText dataset if compute is available
- [ ] Flash attention with Pallas
- [ ] Profiling/Benchmarking and speeding up iteration