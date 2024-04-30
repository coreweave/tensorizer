# Tensorizer serialization via subprocess

If you're using Tensorizer serialization to write checkpoints during training,
you may want to run the serialization concurrently from your training code so
that you can execute your next training step as quickly as possible.  And
because of the Python GIL, it's better to do this in a separate process so that the
serialization doesn't utilize any of the GIL that you'd otherwise use in your training code.

Keep in mind that this is a way to achieve _concurrency_, not instant
snapshotting. The tensors you are checkpointing still need to be kept in memory,
unmodified, for the duration of the serialization process. (Though you may
choose to copy them out of CUDA memory into CPU memory. These tradeoffs are
discussed below.)

Also refer to [PyTorch Multiprocessing best
practices](https://pytorch.org/docs/stable/notes/multiprocessing.html) for more
details about using PyTorch across processes


## Warning about fork() and threads
Be aware that Python `os.fork()` is often not a viable option, as it can be known to cause deadlocks if you have multiple threads. Python 3.12 and above
will [issue a deprecation warning](https://github.com/python/cpython/pull/100229) if you attempt this. Some 3rd-party packages that rely on sockets or file descriptors may also not behave correctly when a process unexpectedly forks. 

A subprocess (fork + exec) is generally safer, but you do not inherently get
shared memory with the calling process. `multiprocessing` has two ways to create
a child process: `spawn` or `forkserver`. `spawn` should always be safe.
`forkserver` can be faster but safety depends on the behavior of modules at
import time. See
https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
for more details.

## If starting from CUDA
Presuming your tensors are in CUDA memory, there are a couple different options.

### Option 1: Communicate the CUDA tensors directly
CUDA tensors can be "shared" to a subprocess very efficiently since it's only communicating a pointer to device memory.

Basically send the CUDA tensors over a `multiprocessing.Queue` to a subprocess that does the serialization. Ensure that the CUDA tensors remain **unmodified** in device memory until the serialization process finishes.

```python
import torch
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM
import torch.multiprocessing as mp

def do_serialize(uri: str, model: torch.nn.Module):
    serializer = TensorSerializer(uri)
    serializer.write_module(model)
    serializer.close()

def my_gpu_model() -> torch.nn.Module:
    model_ref = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to('cuda')
    return model

def main():
    dest = "gpt-j-6B.tensors"
    model = my_gpu_model()

    mp.set_start_method('spawn')
    p = mp.Process(target=do_serialize, args=(dest, model))
    p.start()

    # main process is now free to do other stuff but `model` must remain in CUDA
    # memory until the `p` subprocess finishes

    p.join()


if __name__ == '__main__':
    main()
```

### Option 2: Snapshot CUDA tensors to CPU memory in subprocess before serializing

Once the tensors are in CPU memory, they no longer need to occupy CUDA memory. But the tensors
will now need to occupy CPU memory until they are fully serialized.

Do this by calling `model.to("cpu")` immediately after sending to serializer.

If you like, you can also use some sort of IPC object to communicate back to the
host process when the snapshotting has finished so you know when the CUDA memory
can be released. The below code uses a `Queue`

```python
import torch
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM
import torch.multiprocessing as mp

def do_serialize(uri: str, model: pytorch.nn.Module, snapshot_done: mp.Queue):
    model = model.to('cpu') # Snapshot now 
    snapshot_done.put(True)

    serializer = TensorSerializer(uri)
    serializer.write_module(model)
    serializer.close()

def my_gpu_model() -> torch.nn.Module:
    model_ref = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to('cuda')
    return model

def main():
    dest = "gpt-j-6B.tensors"
    model = my_gpu_model()

    mp.set_start_method('spawn')
    snapshot_done = mp.Queue()
    p = mp.Process(target=do_serialize, args=(dest, model, snapshot_done))
    p.start()

    # main process is now free to do other stuff
    # but `model` must remain in CUDA memory

    snapshot_done.get()
    # Subprocess copied model into CPU memory. Free to release the CUDA-based model
    del model

    # ... do other stuff ...

    if not p.is_alive():
        print('Serialization finished.')

    p.join()


if __name__ == '__main__':
    main()
```

## If starting from CPU memory

Tensors in CPU memory need to moved to shared memory to be communicated with a subprocess. PyTorch `multiprocessing` will do this automatically, but be aware
that a memcpy occurs. You'll also need additional "surge" CPU memory during the duration of the copy of each tensor. PyTorch copies tensors serially, so you need additional memory equal to the size of your largest tensor. This is only used during the memcpy itself. The original non-shared memory is immediately freed thereafter (unless it is also in use by some other tensor)

Depending on how you are constructing your CPU tensor, you may be able to preemptively `tensor.share_memory()` ahead of time, thus saving a memcpy when
passing to the subprocess.

> [!WARNING]
> 
> The main process should avoid modifying tensors while they are being serialized from shared memory, to avoid corrupting the written file. If serializing *with encryption* from shared memory, tensors should additionally not be read again until serialization has finished, as encryption temporarily modifies tensors in-place.
> 
> If concurrent modification or access is necessary, move the tensors out of shared memory and into a copy in the subprocess before serialization. This can be done in the same style shown for snapshotting CUDA tensors in a previous example.

```python
import torch
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM
import torch.multiprocessing as mp

def do_serialize(uri: str, model: torch.nn.Module):
    serializer = TensorSerializer(uri)
    serializer.write_module(model)
    serializer.close()

def my_gpu_model() -> torch.nn.Module:
    model_ref = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return model

def main():
    dest = "gpt-j-6B.tensors"
    model = my_gpu_model()

    mp.set_start_method('spawn')
    
    # this will execute model.share_memory()
    p = mp.Process(target=do_serialize, args=(dest, model))

    p.start()

    # main process is now free to do other stuff
    # but `model` must remain in CPU memory until the `p` subprocess finishes

    p.join()


if __name__ == '__main__':
    main()
```
