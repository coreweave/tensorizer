import sys
import time

kibibyte = 1 << 10
mebibyte = 1 << 20
gibibyte = 1 << 30

def io_test(
    source, read_size=256 * kibibyte, buffer_size=256 * mebibyte
):
    # Read the stream `read_size` at a time.
    buffer = bytearray(read_size)
    total_sz = 0
    start = time.monotonic()
    io = open(source, mode='rb', buffering=buffer_size)
    # io = open(source, mode='rb')
    while True:
        try:
            sz = io.readinto(buffer)
            total_sz += sz
        except OSError:
            break

        if sz == 0:
            break
    end = time.monotonic()

    duration = end-start
    print(f"path={source} total_sz={total_sz}, duration={duration:.3f}s speed={total_sz/duration/1024/1024/1024:.3f}GiB/s buffer_size={buffer_size} read_size={read_size}")

if __name__ == '__main__':
    buffer_size = 256 * mebibyte
    read_size = 256 * mebibyte
    buffer_size = 0
    io_test(sys.argv[1], read_size, buffer_size)