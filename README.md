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


### Pre-Requisites
Note: This repo assumes you have access to a DGXC Sprint/Run:ai cluster and have [kubectl, kubeconfig, and the Run:ai CLI installed](https://docs.nvidia.com/dgx-cloud/run-ai/latest/advanced.html#accessing-the-run-ai-cli).

Export your HuggingFace access token in your terminal:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
echo $HF_TOKEN
YOUR_HF_TOKEN
```

Your HuggingFace token will be referenced in the [run_llama_train.sh](run_llama_train.sh) script to download the Llama Tokenizer:

```bash
# Be sure to export your huggingface token via terminal e.g. export HF_TOKEN=<your HF Token> 
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --local_dir=/torchtitan/datasets/tokenizer/ --hf_token=$HF_TOKEN
```

Install software to run containers like [Docker](https://www.docker.com/get-started/) or [Colima](https://github.com/abiosoft/colima): 

```bash
# Install Docker
brew install --cask docker

# Install Colima
brew install colima

# Start the Colima VM
colima start
```
## Installation
### Clone the repository

```bash
# Git glone the repo
git clone https://github.com/chelseaisaac/torchtitan-runai-distributed.git
# Change directory to the repo
cd torchtitan-runai-distributed/
```

### Use the [Dockerfile](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/main/Dockerfile) to build your container
If you are already familiar with Docker and have a private container registry, you may skip to "Start a Multi-Node Training Run" after you've pushed your pre-built container to your registry using the Dockerfile referenced above. 

For this example we leverage Nvidia's Container Registry to push and pull our pre-built containers from. 
1. First, generate your [personal API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key) from your NGC account and save it somewhere safe. You'll need it in step 2.
2. Log into nvcr.io using your terminal
```bash
# Run the following docker command to start the login process to nvcr.io
docker login nvcr.io

# Enter $oauthtoken as your Username
Username: $oauthtoken

# Copy and paste your Personal API key from NGC from a few steps prior and hit enter 
Password:

# You should see the following message after a successful authentication
WARNING! Your password will be stored unencrypted in /root/username/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credential-stores

Login Succeeded
```

Adjust the docker commands below by inserting your [Organization Name located in your NGC Account's Organization Profile](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#ngc-org-owner-users) (e.g. **xvy2tenvwbmg**) and run the command in your terminal:

```bash
## Container URL is as follows <registry-host>/<namespace>/<repository>:<tag>
## Make sure to update the command with your NGC Org Name e.g. docker build -t nvcr.io/<ORG NAME>/torchtitan-dist:latest .
docker build -t nvcr.io/xvy2tenvwbmg/torchtitan-dist:latest .

## After the container is built, run the following command to push the container to your container registry 
docker push nvcr.io/xvy2tenvwbmg/torchtitan-dist 
```

Add your [NGC Account & Personal API Key](https://docs.nvidia.com/dgx-cloud/run-ai/latest/user-guide.html#credentials) as a Run:ai Credential to pull containers from nvcr.io 

<img width="503" alt="Screenshot 2025-02-25 at 8 50 57 AM" src="https://github.com/user-attachments/assets/883d2bd3-1637-45c9-9be7-fc2380f9bcac" />
1. Navigate to the **Credentials** page illustrated in the image above
2. Click + NEW CREDENTIALS and select Docker registry from the drop down menu. You will be taken to the New credential creation page.
3. Select the Scope for your new NGC credential. The secret will be usable by any workload launched within the scope. For example, if your scope is set at the department level, all workloads launched in any project associated with that department can use the secret, regardless of which user created the credential, or launched the workload.
4. Enter a name and description for the credential. This will be visible to any cluster user.
5. Select New secret.
6. For username, use $oauthtoken.
7. For password, paste your NGC Personal API token.
8. Under Docker Registry URL, enter nvcr.io.
9. Click CREATE CREDENTIALS. Your credentials will now be saved in the cluster and shall be used when you pull a container from your private registry.

### Start a Multi-Node Training Run
Below is an example to submit a Llama 3 8B model on 16 GPUs (2 nodes = 1 primary + 1 worker, 8 GPUs per node) with the Run:ai CLI. In this example, we also pass two environment variables denoted with a '-e' flag that allows you to adjust your configuration file (.toml) to leverage [Llama 8B, 70B, or 405B](https://github.com/chelseaisaac/torchtitan-runai-distributed/tree/main/train_configs) and pass your HuggingFace Token. 

Note: When training the Llama 70B or 405B models using tensor parallelism, it's essential that the model's dimension (8192) is divisible by the number of nodes/shards. For the Llama 70B model, a minimum of 32 GPUs is required. During our tests with the Llama 405B model using 8 nodes (64 GPUs), we encountered an out of memory error.

```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 8 \
        -i nvcr.io/<ORG ID>/torchtitan-dist \
        -e CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"} \
        -e HF_TOKEN=$HF_TOKEN
```
If you'd like to run a training job with a Persistent Volume Claim (PVC) attached, you need to add the --existing-pvc argument along with the pvc name and pvc mount path:
```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 8 \
        -i nvcr.io/<ORG ID>/torchtitan-dist \
        --existing-pvc "claimname=<CLAIM_NAME>,path=<PATH>"  \
        -e CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"} \
        -e HF_TOKEN=$HF_TOKEN
```
You can also verify your PVC's claim name by running the following kubectl command:
```bash
kubectl get pvc
NAME                           STATUS    VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
psa-pvc-project-12345          Bound     pvc-1387dfc8-a6d7-4cb7-894a-59d58cd91f30   10Ti       RWX            zonal-rwx      <unset>                 23d
smcd-pvc-project-67890         Bound     pvc-6a70ec26-c9d5-4e99-94d0-54fe25211b92   10Ti       RWX            zonal-rwx      <unset>                 12d
what-is-pv-project-57wbe       Bound     pvc-2a15ca56-bdb6-4f0a-ab37-96e4be90dec3   10Ti       RWX            zonal-rwx      <unset>                 12d
```

To learn more about PVC's and how to set them up, read the [DGXC Sprint Guide](https://docs.nvidia.com/dgx-cloud/run-ai/latest/user-guide.html#pvc). 

If you wanted to do a single-node training run with 8 GPUs:
```bash
runai submit --name torchtitan \
-i nvcr.io/<ORG ID>/torchtitan \
-g 8 
```
### Upon container's initialization
The script [run_llama_train.sh](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/sarabiap-patch-3/run_llama_train.sh) will execute on start up. View the logs in the UI or use kubectl:

```bash
# Return list of pods
kubectl get pods
NAME                        READY   STATUS      RESTARTS   AGE
pod1-0-0                    1/1     Running     0          16d
pod2-0-0                    1/1     Running     1          3d7h

# Return pod logs
kubectl logs pod1-0-0
=============
== PyTorch ==
=============

NVIDIA Release 25.01 (build 134983853)
PyTorch Version 2.6.0a0+ecf3bae
Container image Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2014-2024 Facebook Inc.
.....................................
```

# To-Do List
- Test with Llama 3 8B — Completed
- Test with Llama 3 70B — Completed
- Test with Llama 3 405b — Not Started (Need 8 nodes or more)

### Modications from the the repository (& why):

The training scripts presented are slightly modified versions of the example scripts that 'torchtitan' provides in their repository.
* [run_llama_train.sh](run_llama_train.sh) - Add environment variables to support distributed PyTorch training with Run:ai.
* [torchtitan/utils.py](torchtitan/utils.py) - Updated seed to be within 32-bit range. The original seed was too large and triggered a runtime error in our environment (DGX Cloud on AWS).
* [torchtitan/train_spec.py](torchtitan/train_spec.py) - Added loss_fn function
* [train.py](train.py) - Changed the labels in train.py. Reshaped the predictions before calculating the loss.
