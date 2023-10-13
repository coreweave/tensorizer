FROM ghcr.io/coreweave/ml-containers/torch:9faf4d7-base-cuda12.0.1-torch2.0.1-vision0.15.2-audio2.0.2

RUN apt-get -qq update && \
     apt-get install -y redis-server lighttpd && \
     apt-get clean
RUN mkdir /app
WORKDIR /app
COPY tensorizer /app/tensorizer
COPY requirements.txt /app/tensorizer
RUN pip3 install -r /app/tensorizer/requirements.txt
ADD examples/benchmark_buffer_size/lighttpd.conf /app/lighttpd.conf
ADD examples/benchmark_buffer_size/benchmark.py /app/benchmark.py
ENTRYPOINT ["/bin/bash",
            "-c",
            "redis-server --daemonize yes >/dev/null & lighttpd -f /app/lighttpd.conf & python /app/benchmark.py" ]
