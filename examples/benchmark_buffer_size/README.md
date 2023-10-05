Buffer Size Benchmarking Framework
==================================
This package contains a benchmarking framework for testing the performance of
`tensorizer` with different transport layers and buffer sizes. Currently, the
script tests the following:

* `redis` transport layer using raw TCP socket (`RedisStreamFile`)
* `https` transport layer using `curl` external process (`CURLStreamFile`)
* `http` transport layer using `curl` external process (`CURLStreamFile`)
* `s3` transport layer which computes authentication and uses `curl` external
  process (`CURLStreamFile`)

It iterates through different buffer sizes, the range given by `--begin` and 
`--end` in powers of 2. For each buffer size, it runs the benchmark for all the
transport layers.

The `buffer_size` has different semantics depending on the transport layer. For
Redis, it's the TCP socket buffer size. For `https`, `http`, and `s3`, it's the
Python buffer size to the `curl` external process.

By default, the `redis` tests are targeted to `localhost` on port `6379`. The
pod definition automatically starts a Redis server on the same pod. We load the
model into the Redis server from the `tensorized` S3 bucket.

Running the Benchmark
---------------------
You should be able to run the benchmark by invoking `kubectl apply -f benchmark.yaml`
from this directory. This will start a Kubernetes Job that runs the benchmark across
10 pods. You can change the number of pods by changing the `parallelism` field in
`benchmark.yaml`.

To look at the benchmark results, you can run `kubectl logs --tail=-1 -l job-name==tensorizer-benchmark-read-size`
which will collect the logs from all the pods and print them out. You can also
look at the logs for individual pods by running `kubectl logs <pod-name>`.

Parameterizing the Benchmark
----------------------------
If you want to test against an external Redis server, you can uncomment the
`--redis` line. We provide a Helm chart in `redis-server.yaml` to deploy a
Redis server in your namespace. You can install it by running `helm install
redis-server redis-server.yaml`.

If you want to test against a different model in the `tensorizer` bucket,
you can provide the `--model` flag. Please note that models larger than 2.7-3B
require the container specs for GPUs to be increased to use a card with more
than 8GB of memory.

Depending on where you deploy your application, you may want to change the
region affinity under `- key: topology.kubernetes.io/region`. This will
ensure that the pods are scheduled in the same region as your application.