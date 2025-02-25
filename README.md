<div align="center">

# Distributed Training Using torchtitan with Run:ai

</div>

In this repo, we will show how to leverage data parallelism using torchtitan and Tensorflow on Kubernetes with Run:ai. We used this code on Torchtitan. [Torchtitan](https://github.com/pytorch/torchtitan) is a repo that showcases PyTorch's latest distributed training features in a clean, minimal codebase. If you are new to torchtitan, please refer to their repo for more information. 

<b>Background:</b>
- With Distributed Data Parallelism (DDP), each GPU gets a copy of the entire model and different batches of data.
- Fully Sharded Data Parallelism (FSDP) shards both model parameters and optimizer states across GPUs/ nodes, reducing memory usage. Each GPU stores a portion of model parameters and optimizer states to save memory.
- With Tensor Parallelism (TP), individual layers of the model are split across multiple GPUs. 

If you would like to learn more about distributed training with Run:ai, please refer to [this Github repo](https://github.com/EkinKarabulut/distributed_training_with_runai). 

# Requirements
1. For this tutorial, we will use a setup consisting of two nodes, each equipped with eight GPUs. However, you can scale up by adding more nodes or GPUs to suit your specific requirements.

2. Make sure you have access to a Run:ai environment with version 2.10 or a later release. (Run:ai provides a comprehensive platform for managing and scaling deep learning workloads on Kubernetes).

3. Prepare an image registry, such as a NGC or Docker Hub account, where you can store your custom container images for distributed training.<br>

4. Create a Hugging Face account and agree to Meta Llama 3.1 Community License agreement while signed into Hugging Face account. Generate a Hugging Face read access token in [account settings](https://huggingface.co/settings/tokens). It’s required to access the Llama3.1-8B model. (`torchtitan` currently supports training Llama 3.1 (8B, 70B, 405B) out of the box.)


### Pre-work

Export your HuggingFace access token in your terminal:

```
export HF_TOKEN=<YOUR_HF_TOKEN>
echo $HF_TOKEN
YOUR_HF_TOKEN
```

Your HuggingFace token will be referenced in the [run_llama_train.sh](run_llama_train.sh) script to download the Llama Tokenizer:

```
# Be sure to export your huggingface token via terminal e.g. export HF_TOKEN=<your HF Token> 
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=$HF_TOKEN
```

Install software to run containers like [Docker](https://www.docker.com/get-started/) or [Colima](https://github.com/abiosoft/colima): 

```
# Install Docker
brew install --cask docker
# Install Colima
brew install colima
```
## Installation
### Clone the repository

```
# Git glone the repo
git clone https://github.com/chelseaisaac/torchtitan-runai-distributed.git
# Change directory to the repo
cd torchtitan-runai-distributed/
```

### Use the Dockerfile to build your container
If you want to create your own image, you can edit your code, create your image and push the image to your image registry with the following commands:

```
docker build -t nvcr.io/<ORG ID>/torchtitan-dist .
docker push nvcr.io/<ORG ID>/torchtitan-dist 
```

### Start a multi-node training run
Llama 3 8B model on 16 GPUs (2 nodes = 1 primary + 1 worker, 8 GPUs per node). We pass an environment variable (-e) that allows you to adjust your configuration file (.toml) to leverage [Llama 8B, 70B, or 405B](https://github.com/chelseaisaac/torchtitan-runai-distributed/tree/main/train_configs) and your HuggingFace Token.

```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 8 \
        -i nvcr.io/<ORG ID>/torchtitan-dist \
        -e CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"} \
        -e HF_TOKEN=$HF_TOKEN
```
If you'd like to run the same job with a PVC attached, here's the command:

```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 8 \
        -i nvcr.io/<ORG ID>/torchtitan-dist \
        --existing-pvc "claimname=<CLAIM_NAME>,path=<PATH>"  \
        -e CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"} \
        -e HF_TOKEN=$HF_TOKEN
```

If you wanted to do a single-node training run with 8 GPUs:
```bash
runai submit --name torchtitan \
-i nvcr.io/<ORG ID>/torchtitan \
-g 8 
```

# To-Do List
- Test with Llama3-405b model with FSDP and TP

### Modications from the the repository (& why):

The training scripts presented are slightly modified versions of the example scripts that 'torchtitan' provides in their repository.
* [run_llama_train.sh](run_llama_train.sh) - Add environment variables to support distributed PyTorch training with Run:ai.
* [torchtitan/utils.py](torchtitan/utils.py) - Updated seed in to be within 32-bit range. The original seed was too large and triggered a runtime error in our environment (DGX Cloud on AWS).
* [torchtitan/train_spec.py](torchtitan/train_spec.py) - Added loss_fn function
* [train.py](train.py) - Changed the labels in train.py. Reshaped the predictions before calculating the loss.
