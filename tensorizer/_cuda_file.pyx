# cython: language_level=3

from libc.errno cimport errno
from libc.stdio cimport printf
from libc.stdint cimport int64_t
from posix.mman cimport mmap, munmap, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS, MAP_FAILED
from posix.time cimport clock_gettime, CLOCK_MONOTONIC, timespec
from posix.unistd cimport pread

ctypedef char* char_ptr

cdef extern from "cuda_runtime_api.h":
    ctypedef unsigned int cudaError_t
    ctypedef unsigned int cudaHostRegisterFlags
    ctypedef unsigned int cudaMemcpyKind
    ctypedef unsigned int cudaStream_t

    cdef unsigned int cudaMemcpyHostToDevice = 1
    cudaError_t cudaHostRegister(void *ptr, size_t size, cudaHostRegisterFlags flags) nogil
    cudaError_t cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind) nogil
    cudaError_t cudaMemcpyAsync(void *dst, void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil


cdef void* create_anonymous_mmap(size_t length) nogil:
    # Using MAP_PRIVATE | MAP_ANONYMOUS to create an anonymous mapping
    # Pass -1 as the file descriptor and 0 as offset for anonymous mapping
    cdef void* addr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    
    if addr == MAP_FAILED:
        with gil:
            raise MemoryError("mmap failed")

    return addr  # Cast the address to a Python object before returning

cdef struct PerfTimer:
    timespec start_time
    int64_t elapsed_ns

cdef void StartTimer(PerfTimer* timer) noexcept nogil:
    clock_gettime(CLOCK_MONOTONIC, &timer.start_time)

cdef void StopTimer(PerfTimer* timer) noexcept nogil:
    cdef timespec end_time
    clock_gettime(CLOCK_MONOTONIC, &end_time)

    cdef int64_t sec = end_time.tv_sec - timer.start_time.tv_sec
    cdef int64_t nsec = end_time.tv_nsec - timer.start_time.tv_nsec
    timer.elapsed_ns += sec * 1000000000 + nsec

cdef PerfTimer _NewPerfTimer() noexcept nogil:
    cdef PerfTimer timer
    timer.elapsed_ns = 0
    return timer


def allocate_buffer(int buffer_size):
    cdef PerfTimer timer = _NewPerfTimer()
    cdef cudaError_t err
    cdef void* buffer

    with nogil:
        StartTimer(&timer)

        buffer = create_anonymous_mmap(buffer_size)
        err = cudaHostRegister(buffer, buffer_size, 0)

        StopTimer(&timer)

        if err != 0:
            raise ValueError('Cuda error: %d' % err)

        printf("Elapsed time: %ldns\n", timer.elapsed_ns)

    return <unsigned long>(buffer)

def deallocate_buffer(unsigned long buffer, int buffer_size):
    err = munmap(<void *>(buffer), buffer_size)
    if err != 0:
        raise ValueError('munmap failed: %d' % err)

def copy_to_device(int fd, unsigned long device_ptr, ssize_t size, unsigned long fd_offset, unsigned long pinned_buffer, unsigned int pinned_buffer_size):
    # expectations
    # fd is a file descriptor
    # device_ptr is a pointer to cuda device memory
    # buffer is a value returned from allocate_buffer
    cdef char* _device_ptr = <char*>(device_ptr)

    cdef ssize_t read_bytes
    cdef ssize_t to_read_size
    cdef cudaError_t err
    cdef ssize_t orig_size = size

    cdef PerfTimer readTimer = _NewPerfTimer()
    cdef PerfTimer copyTimer = _NewPerfTimer()

    pinned_buffer_size = pinned_buffer_size>>1
    cdef char_ptr[2] pinned_buffers_flipflop = [<char*>(pinned_buffer), <char*>(pinned_buffer) + (pinned_buffer_size)]
    cdef int flipflop = 0

    with nogil:
        while size > 0:
            to_read_size = min(size, pinned_buffer_size)
            StartTimer(&readTimer)
            read_bytes = pread(fd, <void *>(pinned_buffers_flipflop[flipflop]), to_read_size, fd_offset)
            StopTimer(&readTimer)
            if read_bytes != to_read_size:
                # TODO: warning?
                raise ValueError('Read %d bytes, expected %d. errno=%d' % (read_bytes, to_read_size, errno))

            StartTimer(&copyTimer)
            err = cudaMemcpyAsync(_device_ptr, <void *>(pinned_buffers_flipflop[flipflop]), read_bytes, cudaMemcpyHostToDevice, 0)
            StopTimer(&copyTimer)
            if err != 0:
                raise ValueError('Cuda error in cudaMemcpy: %d' % err)

            fd_offset += read_bytes
            _device_ptr += read_bytes
            size -= read_bytes
            flipflop = not flipflop

        printf("Read time: %ldns, %.3f GB/s\n", readTimer.elapsed_ns, orig_size / readTimer.elapsed_ns)
        printf("Copy time: %ldns, %.3f GB/s\n", copyTimer.elapsed_ns, orig_size / copyTimer.elapsed_ns)