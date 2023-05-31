FROM ghcr.io/coreweave/ml-containers/torch:bb02bee-nccl-cuda12.0.1-nccl2.17.1-1-torch2.0.0-vision0.15.1-audio2.0.1

RUN pip install transformers tensorizer==1.1.0 accelerate safetensors==0.3.1

RUN mkdir /app
ADD deserialize_benchmark.py /app/deserialize_benchmark.py
ADD save_models.py /app/save_models.py
