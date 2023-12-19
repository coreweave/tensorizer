FROM ubuntu:20.04

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive \
      apt-get -qq install --no-install-recommends -y bash python3 && \
    apt-get clean

WORKDIR /app
COPY shared-fs-benchmark.py ./

ENTRYPOINT [ "python3", "shared-fs-benchmark.py" ]
