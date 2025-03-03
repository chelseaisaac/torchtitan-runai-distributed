<div align="center">

# Distributed Training Using torchtitan with Run:ai

</div>

# Table of Contents
[Overview](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/main/README.md#overview)  
[Pre-Requisites](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/main/README.md#pre-requisites)  
[Using the Repository](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/main/README.md#using-the-repository)  
[To-Do List & Updates to torchtitan Repository](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/main/README.md#updates--to-do-list)

# Overview
This repository contains code we have implemented and tested to demonstrate how to effectively leverage data parallelism with torchtitan on a Kubernetes cluster utilizing Run:ai. [Torchtitan](https://github.com/pytorch/torchtitan) is a repository that highlights the latest distributed training features in PyTorch, offering a clean and minimalistic codebase. If you are new to Torchtitan, we encourage you to visit the [torchtitan GitHub repository](https://github.com/pytorch/torchtitan) for comprehensive information and resources.

If you would like to learn more about distributed training with Run:ai, please refer to [this Github repo](https://github.com/EkinKarabulut/distributed_training_with_runai). 

## Key torchtitan Features:
- With Distributed Data Parallelism (DDP), each GPU gets a copy of the entire model and different batches of data.
- Fully Sharded Data Parallelism (FSDP) shards both model parameters and optimizer states across GPUs/ nodes, reducing memory usage. Each GPU stores a portion of model parameters and optimizer states to save memory.
- With Tensor Parallelism (TP), individual layers of the model are split across multiple GPUs.
You can read more about torchtitan's features [here](https://github.com/pytorch/torchtitan?tab=readme-ov-file#key-features-available).

## Background
1. For this tutorial, we will use a setup consisting of two nodes, each equipped with eight GPUs. However, you can scale up by adding more nodes or GPUs to suit your specific requirements.

2. Make sure you have access to a Run:ai environment with version 2.10 or a later release. (Run:ai provides a comprehensive platform for managing and scaling deep learning workloads on Kubernetes).

3. Prepare an image registry, such as a NGC or Docker Hub account, where you can store your custom container images for distributed training.<br>

4. Create a Hugging Face account and agree to Meta Llama 3.1 Community License agreement while signed into Hugging Face account. Generate a Hugging Face read access token in [account settings](https://huggingface.co/settings/tokens). It’s required to access the Llama3.1-8B model. (`torchtitan` currently supports training Llama 3.1 (8B, 70B, 405B) out of the box.)


# Pre-Requisites
Note: This repo assumes you have access to a DGXC Sprint/Run:ai cluster and have [kubectl, kubeconfig, and the Run:ai CLI installed](https://docs.nvidia.com/dgx-cloud/run-ai/latest/advanced.html#accessing-the-run-ai-cli).

## Install kubectl:
```bash
# Install on macOS
brew install kubectl

# View version
kubectl version --client

# OPTIONAL SECTION 
# Set kubectl as an alias fo to your shell configuration file (e.g. ~/.bashrc, ~/.bash_profile, ~/.zshrc, etc.)
nano ~/.zshrc

# Append the following on a new line in the file
alias k='kubectl'

# Save your shell configuration file
ctrl + o
Enter

# Exit the editor
ctrl + x

# Apply the changes to your shell configuration file
source ~/.zshrc
```

### Save your kubeconfig:
```bash
# Create a folder named .kube
mkdir ~/.kube

# Change directory to .kube
cd ~/.kube

# Create your kubeconfig file
touch kubeconfig

# Ask your cluster admin for your kubeconfig and paste the contents into the editor
ctrl + v or cmd + v

# Save your kubeconfig
ctrl + o
Enter

# Exit the editor
ctrl + x

# View your kubeconfig in the .kube folder
ls
```

## Download the Run:ai CLI
1. In your web browser, navigate to your Run:ai login page by entering your Run:ai URL e.g. https://app.run.ai
2. Select CONTINUE WITH SSO
3. In the top right of the Run:ai interface, click the '**?**' and select **Researcher Command Line Interface**
4. Select your operating system
5. Paste the **wget** command in terminal
6. After installation, run the following command in terminal:
```bash
# Change the permissions to make runai an executable
chmod +x runai

# Move runai file to your ~/local/bin directory
sudo mv runai /usr/local/bin/runai

# Authenticate your CLI
runai login
Go to the following link in your browser:
<Copy and paste this run.ai URL>

# Paste the verification code from your web browser back into your terminal
Enter verifcation code: #########

INFO[0248] Logged in successfully
```   

## Install software to run containers like [Docker](https://www.docker.com/get-started/) or [Colima](https://github.com/abiosoft/colima): 

```bash
# Install Docker
brew install --cask docker

# Install Colima
brew install colima

# Start the Colima VM
colima start
```

## Download the Llama Tokenizer
We've included the tokenizer.model in this repository so you don't have to download it from HuggingFace. See the Deprecated Section at the end of the repo.  

## Nvidia GPU Cloud (NGC) Setup
If you are already familiar with Docker and have a private container registry, you may skip this section to **[Start a Multi-Node Training Run](#start-a-multi-node-training-run)** after you've pushed your pre-built container to your registry using the Dockerfile referenced above. 

### Generate Personal Key
For this example, we leverage Nvidia's Container Registry to push and pull our pre-built containers from. Alternatively, you can use docker.io via Docker Hub.
1. First, generate your [personal API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key) from your NGC account. Enter the following fields: <br>
        Key Name <br>
        Expiration <br>
        Services Included: <br> 
        Secrets Manager <br>
        NGC Catalog <br>
        Private Registry <br>
        Cloud Functions <br>

Click **Generate Personal Key** and save it somewhere safe. You'll need it in the following step. <br><br>
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
### Install the NGC CLI
The NGC CLI installation instructions can be found [here](https://org.ngc.nvidia.com/setup/installers/cli).

```bash
## Download the CLI package based on your operating system
## For example, ARM64 MacOs
curl -LO https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.60.2/files/ngccli_mac_arm.pkg

## Check the installer's SHA256 hash to ensure the file wasn't corrupted during download
shasum -a 256 ngccli_mac_arm.pkg

## Verify the output of the SHA256 checksum
c3733a4f8974a28b486a965be31e1ae7f1c7b6af68b10a2490766b5a17cca498

## Run the installer
sudo installer -pkg ngccli_mac_arm.pkg -target /usr/local

## Configure your NGC CLI config
ngc config set

## You'll be prompted to enter the following details:
Enter API key [********]. Choices: [<VALID_APIKEY>, 'no-apikey']: <Enter Personal API Key>
Enter CLI output format type [ascii]. Choices: ['ascii', 'csv', 'json']: <Enter CLI output format>
Enter org [###########]. Choices: ['[###########']: <Enter Unique Org ID>
Enter team [no-team]. Choices: ['no-team']: <enter no-team or team unless you have this configured>
Enter ace [no-ace]. Choices: ['no-ace']: <enter no-ace unless you this configured>

## Output results upon completing configuration
Validating configuration...
Successfully validated configuration.
Saving configuration...
Successfully saved NGC configuration to ~/username/.ngc/config
```

# Using the Repository
## Git Clone the Repository

```bash
# Git glone the repo
git clone https://github.com/chelseaisaac/torchtitan-runai-distributed.git
# Change directory to the repo
cd torchtitan-runai-distributed/
```

## Build Your Container
Leverage the [Dockerfile](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/main/Dockerfile) to build your container. You make adjustments as needed to support your unique environment. To kickoff the container build, adjust the docker commands below by inserting your [Organization Name located in your NGC Account's Organization Profile](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#ngc-org-owner-users) (e.g. **xvy2tenvwbmg**) and run the command in your terminal:

```bash
## Container URL is as follows <registry-host>/<namespace>/<repository>:<tag>
## Make sure to update the command with your NGC Org Name e.g. docker build -t nvcr.io/<ORG NAME>/torchtitan-dist:latest .
docker build -t nvcr.io/xvy2tenvwbmg/torchtitan-dist:latest --platform linux/amd64 .

## After the container is built, run the following command to push the container to your container registry 
docker push nvcr.io/xvy2tenvwbmg/torchtitan-dist 
```

## Create a NGC Credential in Run:ai
Add your [NGC Account & Personal API Key](https://docs.nvidia.com/dgx-cloud/run-ai/latest/user-guide.html#credentials) as a Run:ai Credential to pull containers from nvcr.io 

<img width="503" alt="Screenshot 2025-02-25 at 8 50 57 AM" src="https://github.com/user-attachments/assets/362465f1-39fd-4e85-b3f9-d6057f747ac6" /><br>

1. Navigate to the **Credentials** page illustrated in the image above (steps #1 - #3).
2. Click + NEW CREDENTIALS and select Docker registry from the drop down menu. You will be taken to the New credential creation page.
3. Select the Scope for your new NGC credential. The secret will be usable by any workload launched within the scope. For example, if your scope is set at the department level, all workloads launched in any project associated with that department can use the secret, regardless of which user created the credential, or launched the workload.
4. Enter a name and description for the credential. This will be visible to any cluster user.
5. Select New secret.
6. For username, use $oauthtoken.
7. For password, paste your NGC Personal API token.
8. Under Docker Registry URL, enter nvcr.io.
9. Click CREATE CREDENTIALS. Your credentials will now be saved in the cluster and shall be used when you pull a container from your private registry.

## Start a Multi-Node Training Run
Below is an example command to train the Llama 3.1 8B model on 16 GPUs (2 nodes = 1 primary + 1 worker, 8 GPUs per node) with the Run:ai CLI. In this example, we also pass two environment variables denoted with a '-e' flag that allows you to adjust your configuration file (.toml) to leverage [Llama 8B, 70B, or 405B](https://github.com/chelseaisaac/torchtitan-runai-distributed/tree/main/train_configs) and pass your HuggingFace Token. 

Note: When training the Llama 70B or 405B models using tensor parallelism, it's essential that the model's dimension (8192) is divisible by the number of nodes/shards. For the Llama 70B model, a minimum of 32 GPUs is required. During our tests with the Llama 405B model using 8 nodes (64 GPUs), we encountered an out of memory error.

**Llama 8B Example**
```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 8 \
        -i nvcr.io/<ORG NAME>/torchtitan-dist \
        -e CONFIG_FILE="./train_configs/llama3_8b.toml"
```

**Single Node (8 GPU) Example**
```bash
runai submit --name torchtitan \
        -i nvcr.io/<ORG NAME>/torchtitan \
        -g 8 \
        -e CONFIG_FILE="./train_configs/llama3_8b.toml"
```

**Llama 405B Example** <br>_(Note: We've added additional environment variables to improve redundancies in the event you encounter pods restarts or throttling)_
```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=15 -g 8 \
        -i nvcr.io/<ORG NAME>/torchtitan-dist \
        -e CONFIG_FILE="./train_configs/llama3_405b.toml" \
        -e RDZV_TIMEOUT=3600 \
        -e MAX_RESTARTS=10 \
        -e HF_HUB_ETAG_TIMEOUT=500 \
        -e HF_HUB_DOWNLOAD_TIMEOUT=120
```

**Upon Pod Initialization** <br>The script [run_llama_train.sh](https://github.com/chelseaisaac/torchtitan-runai-distributed/blob/sarabiap-patch-3/run_llama_train.sh) will execute on start up.

![Screenshot 2025-02-27 at 2 06 07 PM](https://github.com/user-attachments/assets/5f92c87b-cd1b-4ea4-97dc-41d2098ee2a6)


## Create a Persistent Volume Claim (PVC) in Run:ai
A Persistent Volume Claim (PVC) is a request for dedicated storage that allows your data to persist beyond the lifecycle of a pod. It ensures that the data remains accessible to containers even after the pod is terminated. To learn more about PVC's and how to set them up, read the [Run:ai on DGX Cloud Guide](https://docs.nvidia.com/dgx-cloud/run-ai/latest/user-guide.html#pvc). 

<img width="373" alt="Screenshot 2025-02-25 at 10 08 01 AM" src="https://github.com/user-attachments/assets/2111fb2a-3565-4276-8702-9c18abdc635f" />

1. After selecting PVC, you will be taken to the New data source creation page.
2. Set a Scope for the PVC, and enter a name and description.
3. Fill out the Data mount section of the form:
4. Select a Storage class. Be sure to review the [DGX Cloud recommended storage classes](https://docs.nvidia.com/dgx-cloud/run-ai/latest/user-guide.html#user-guide-recommended-storage-classes).
5. Select the access mode configuration for the PVC - either read/write by one node, read only by many nodes, or read/write by many nodes.
6. Specify a claim size to ensure a minimum capacity for the PVC.
7. Choose the Filesystem option as the Volume mode (Block is unsupported).
8. Specify a Container path to define what path the PVC will be accessible from in a running job.
9. (Optional) In the Restrictions pane, you can use the toggle switch to make the storage read-only if desired.
10. Click CREATE DATA SOURCE. You will be taken to the Data sources overview page, where you can view your new PVC data source.

**Persistent Volume Claim Example**<br>
If you'd like to run a training job with a **Persistent Volume Claim (PVC)** attached, you need to add the _--existing-pvc_ argument along with the pvc name and pvc mount path:
```bash
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 8 \
        -i nvcr.io/<ORG NAME>/torchtitan-dist \
        --existing-pvc "claimname=<CLAIM_NAME>,path=<PATH>"  \
        -e CONFIG_FILE="./train_configs/llama3_8b.toml"
```

You can also verify your PVC's claim name by running the following kubectl command:
```bash
kubectl get pvc
NAME                           STATUS    VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
alpha-pvc-project-12345        Bound     pvc-00000000-0000-0000-0000-000000000000   10Ti       RWX            zonal-rwx      <unset>                 23d
beta-pvc-project-67890         Bound     pvc-00000000-0000-0000-0000-000000000000   10Ti       RWX            zonal-rwx      <unset>                 12d
gamma-pv-project-12345         Bound     pvc-00000000-0000-0000-0000-000000000000   10Ti       RWX            zonal-rwx      <unset>                 12d
```

# View Logs

## Using Run:ai CLI

```bash
# Return list of jobs
runai list jobs
Showing jobs for project my-project
NAME       STATUS   AGE  NODE            IMAGE                                    TYPE     PROJECT                                 USER                   GPUs Allocated (Requested)  PODs Running (Pending)  SERVICE URL(S)
nccl-test  Deleted  68d  -               pytorch/pytorch:latest                   Mpi      my-project                              user@company.com       16.00          (16.00)      16            (0)

# Return logs
runai logs nccl-test
<Your Logs Here>         
```

## Using kubectl

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

## View Logs in UI

<img width="2480" alt="Screenshot 2025-02-25 at 11 17 43 AM" src="https://github.com/user-attachments/assets/3ab3ee78-b6dd-48ef-b026-577d9748c204" />

## Tensorboard

### Update toml Config File

```bash
# Update [metrics]
[metrics]
log_freq = 10
enable_tensorboard = true
save_tb_folder = "tb"
enable_wandb = false
```

### Portforward Container
```bash
# In a new terminal window, run
runai port-forward distributed-training-pytorch --port 8888:8888 OR kubectl port-forward pod/<pod-name> <local-port>:<pod-port>

# You should see the following output
open access point(s) to service from localhost:8888
Forwarding from 127.0.0.1:8888 -> 8888
Forwarding from [::1]:8888 -> 8888
Handling connection for 8888 -- accessed http://localhost:8888 via browser
```

### Tensorboard Logs
Tensorboard logging shall be saved in ```./outputs/tb/<filename>```. Verify the exact directory by viewing your logs. Save this path for later.

```bash
[rank0]:[titan] 2025-02-27 19:35:04,661 - root - INFO - TensorBoard logging enabled. Logs will be saved at ./outputs/tb/20250227-1935
```

### Launch Tensorboard
```bash
# SSH into the container
runai bash distributed-training-pytorch

# Launch Tensorboard
tensorboard --logdir=./outputs/tb/20250227-1935 --port=8888
..................................................................
..................................................................
NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

# Copy the URL below into your browser to launch the UI
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.18.0 at http://localhost:8888/ (Press CTRL+C to quit)
```
![Screenshot 2025-02-27 at 11 53 28 AM](https://github.com/user-attachments/assets/ba399390-e6de-4d13-af12-82ec741648fa)

## Weights & Biases
[Under Construction]

## Nsight Systems
NVIDIA Nsight™ Systems is a system-wide performance analysis tool designed to visualize an application’s algorithms, identify the largest opportunities to optimize, and tune to scale efficiently across any quantity or size of CPUs and GPUs, from large servers to our smallest systems-on-a-chip (SoCs). If you are using the DockerFile from this repo, Nsight Systems will already be preloaded in the container. Read more about Nsight Systems [here](https://developer.nvidia.com/nsight-systems).

nsys allows for various interactive CLI command sequences to trigger data collection. For example:

**Run application, begin collection manually, stop run manually** 
(Note: You may want to introduce a delay to bypass the warm up period) 

```bash
# Include the Nsight Systems launch command within your script
nsys launch \
torchrun --nproc_per_node=................

# SSH into your container
runai bash <job_nam>

# Go to your desired directory. If you want to save the output file, go to your PVC.
cd
cd </desired_directory>

# Manually kick-off data collection
nsys start

# Manually stop data collection
nsys stop
Generating '/tmp/nsys-report.qdstrm'
[1/1] [========================100%] report.nsys-rep
Generated:
    </desired_directory>/report.nsys-rep

# Generate the results of the report
nsys stats report.nsys-rep
Generating SQLite file report2.sqlite from report.nsys-rep

** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)                 Name               
 --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------
     ##.#      ###########      #####     ######.#      #####.#      ####   ########    #######.#  cudaKernel                

```

**Profile a Python script that uses CUDA**
(Note: This example launches a Python script and starts profiling 60 seconds after the launch, tracing CUDA, cuDNN, cuBLAS, OS runtime APIs, and NVTX. Do not collect CPU sampling information or thread scheduling information. Profile any child processes. Generate the output file as <my_file>.nsys-rep in the current working directory.)
```bash
# Include the Nsight Systems launch command with in your script
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
--delay=60 --sample=none --cpuctxsw=none -o <my_file> \
python train.py
```

**Run Application, Start / Stop Collection using cudaProfilerStart/Stop**
(Note: Create interactive CLI process and set it up to begin collecting as soon as a cudaProfileStart() is detected. Launch application for default analysis, sending application output to the terminal. Stop collection at next call to cudaProfilerStop, when the user calls nsys stop, or when the root process terminates. Generate the report#.nsys-rep in the default location.)
```bash
# Include the Nsight Systems launch command with in your script
nsys start -c cudaProfilerApi
nsys launch -w true <application> [application-arguments]
```

To read more, go to the [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

# To-Do List & Updates to torchtitan Repository 
- Test with Llama 3 8B — Completed
- Test with Llama 3 70B — Completed
- Test with Llama 3 405b — In-Progress (Note: Need **at least 16 nodes** or more)

### Modications from the torchtitan repository (& why):

The training scripts presented are slightly modified versions of the example scripts that 'torchtitan' provides in their repository.
* [run_llama_train.sh](run_llama_train.sh) - Add environment variables to support distributed PyTorch training with Run:ai.
* [torchtitan/utils.py](torchtitan/utils.py) - Updated seed to be within 32-bit range. The original seed was too large and triggered a runtime error in our environment (DGX Cloud on AWS).
* [torchtitan/train_spec.py](torchtitan/train_spec.py) - Added loss_fn function
* [train.py](train.py) - Changed the labels in train.py. Reshaped the predictions before calculating the loss.

# Deprecated Section(s)

## Export HuggingFace Token
To read more about [HF access tokens, go here](https://huggingface.co/docs/hub/en/security-tokens). 

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
