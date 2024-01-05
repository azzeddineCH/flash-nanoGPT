################### Running on single TPU chip ############################

gcloud compute tpus tpu-vm create flash-nano-gpt-tpu \
  --zone=us-central1-f \
  --accelerator-type=v2-8 \
  --version=tpu-ubuntu2204-base

# gcloud compute tpus tpu-vm ssh flash-nano-gpt-tpu \
#  --zone=us-central1-f

gcloud compute tpus tpu-vm ssh flash-nano-gpt-tpu \
  --zone=us-central1-f \
  --command="
    git clone https://github.com/azzeddineCH/flash-nanoGPT.git
    cd flash-nanoGPT
    pip install tyro
    pip install orbax-checkpoint
    pip install tiktoken
    pip install tensorflow
    pip install flax
    pip install optax
    pip install git+https://github.com/deepmind/jmp
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install wandb
    pip install datasets
    sudo apt install tmux
  "

gcloud compute tpus tpu-vm ssh flash-nano-gpt-tpu \
  --zone=europe-west4-a \
  --worker=all \
  --command="
    cd flash-nanoGPT
    tmux new -d -s nano-gpt '
     export GPT_CONFIG_PATH='yaml/nano-gpt.yaml'
     export WANDB_API_KEY='...'
     git pull
     python3 train.py --grad_accum_steps 32
    "

################### Running on TPU VM ############################


gcloud compute tpus tpu-vm create flash-nano-gpt-tpu-vm \
  --zone=europe-west4-a \
  --accelerator-type=v3-32  \
  --version=tpu-ubuntu2204-base \
    --preemptible

# gcloud compute tpus tpu-vm delete flash-nano-gpt-tpu-vm  \
#  --zone=europe-west4-a \

gcloud compute tpus tpu-vm ssh flash-nano-gpt-tpu-vm \
  --zone=europe-west4-a \
  --worker=all \
  --command="
    git clone https://github.com/azzeddineCH/flash-nanoGPT.git
    cd flash-nanoGPT
    pip install tyro
    pip install orbax-checkpoint
    pip install tiktoken
    pip install tensorflow
    pip install flax
    pip install optax
    pip install git+https://github.com/deepmind/jmp
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install wandb
    pip install datasets
    sudo apt install tmux
  "

gcloud compute tpus tpu-vm scp test.py flash-nano-gpt-tpu-vm: \
  --worker=all \
  --zone=europe-west4-a

gcloud compute tpus tpu-vm ssh flash-nano-gpt-tpu-vm \
  --zone=europe-west4-a \
  --worker=all \
  --command="
  python3 test.py"

gcloud compute tpus tpu-vm ssh flash-nano-gpt-tpu-vm\
  --zone=europe-west4-a \
  --worker=all \
  --command="
    cd flash-nanoGPT
    tmux new -d -s nano-gpt '
     export GPT_CONFIG_PATH='yaml/train-nano-gpt.yaml'
     export WANDB_API_KEY='...'
     git pull
     python3 train.py --no-save-checkpoint --no-amp
    '
  "