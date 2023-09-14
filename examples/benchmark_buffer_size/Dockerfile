FROM ghcr.io/coreweave/ml-containers/torch:9faf4d7-base-cuda12.0.1-torch2.0.1-vision0.15.2-audio2.0.2

RUN mkdir /app
COPY tensorizer /app/tensorizer
ADD examples/benchmark_buffer_size/benchmark.py /app/benchmark.py
ENTRYPOINT ["python3", "/app/benchmark.py"]
