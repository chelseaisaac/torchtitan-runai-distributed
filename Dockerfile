FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN git clone https://github.com/pytorch/torchtitan.git && \
    cd torchtitan && \
    pip install -r requirements.txt && \
    pip install . && \
    pip install --upgrade torch

#Set the working directory
WORKDIR /app

#Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY train.py run_llama_train.sh /app/
COPY train_configs/ /app/train_configs/
COPY tests/ /app/tests/
COPY assets/ /app/assets/
COPY scripts/ /app/scripts/
COPY torchtitan/ /app/torchtitan/


#Run the bash file
RUN chmod u+x run_llama_train.sh
CMD ["./run_llama_train.sh"]