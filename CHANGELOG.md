# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tensors with long dimensions (≥ `2 ** 32` elements in a single dimension)
  can now be serialized and deserialized

### Fixed

- `tensorizer.torch_compat` can now serialize and deserialize tensors that have
  storages with sizes ≥ `2 ** 32`

## [2.11.1] - 2025-08-05

### Fixed

- Fixed an incompatibility between `tensorizer.torch_compat` and `torch<2.7.1`
  - Thank you to `stuart-mv` for catching this

## [2.11.0] - 2025-08-04

### Added

- `tensorizer.torch_compat` is a new module for using `tensorizer` as a backend
  for handling tensor data during standard `torch.save` and `torch.load` calls
  - To use `tensorizer` as a backend for `torch.save`,
    wrap the call in the `tensorizer_saving` context manager
    - The file created must then be loaded using `tensorizer_loading`
  - To use `tensorizer` as a backend for `torch.load`,
    wrap the call in the `tensorizer_loading` context manager
    - The file to load must have been created using `tensorizer_saving`

## [2.10.1] - 2025-06-27

### Fixed

- `TensorDeserializer` objects now respect CUDA devices chosen with the
  `torch.device()` and `torch.cuda.device()` context managers
  - Previously, the `device` parameter to the `TensorDeserializer` constructor
    was the only way to choose between multiple CUDA devices
    (e.g. `cuda:0`, `cuda:1`)
  - Now, when a device with no index is specified, such as
    `device=torch.device("cuda")`, `device="cuda"`, or even `device=None`,
    the local device context is used to select the device
  - `torch.cuda.set_device()` and `torch.set_default_device()`
    are also supported
  - This doesn't add support for detecting CPU contexts such as
    `torch.device("cpu")` without explicitly specifying `device="cpu"`
    - This is for backwards compatibility, as the `TensorDeserializer`
      default device has always been mandated to be CUDA whenever CUDA
      is available, and `torch` does not provide a public interface to
      reliably disambiguate an intentional `torch.device("cpu")` context
      from the global default
  - Lazy-loaded tensors will use the active device as of the time
    they are actually loaded
- `TensorDeserializer` objects no longer fail to open file-like objects
  whose `fileno()` methods raise `io.UnsupportedOperation`

## [2.10.0] - 2025-06-09

### Added

- `stream_io.open_stream()` now respects Boto3's configuration files
  and environment variables when searching for object storage credentials to use

### Fixed

- `stream_io.open_stream()` now uses virtual-hosted-style
  bucket addressing for the `cwobject.com` and `cwlota.com` endpoints
- `stream_io.open_stream()` now allows the `use_https` entry of `.s3cfg`
  configuration files to fill in its `force_http` parameter if `force_http` is
  not explicitly specified as `True` or `False`
- `TensorSerializer` no longer throws an error when attempting to serialize
  very large tensors on some non-Linux platforms
- Object storage uploads managed by `stream_io.open_stream()` now finalize
  correctly on Python 3.12+ even without an explicit call to their `close()`
  method
  - A fix for this was originally implemented in release 2.7.2,
    but it only worked for Python versions below 3.12

## [2.9.3] - 2025-05-09

### Changed

- `stream_io.open_stream()` now defaults to authenticating with signature
  version 4 rather than signature version 2 when no `s3_signature_version`
  is specified for reads from most object storage endpoints

## [2.9.2] - 2025-02-20

### Fixed

- Fixed compatibility with `numpy>=2.0.0`
  - Calls to the removed `numpy.product` function now use `numpy.prod` instead

## [2.9.1] - 2024-11-27

### Fixed

- `TensorSerializer` no longer sometimes fails to serialize very large
  1-dimensional tensors with multibyte `dtype`s
- `RedisStreamFile.readable()` and `RedisStreamFile.seekable()` now correctly
  return `True`

## [2.9.0] - 2024-04-17

### Added

- Multiple file readers during deserialization ([#87])
  - Controlled by the new `num_readers` `int` parameter
    to the `TensorDeserializer` constructor
  - Files capable of having multiple readers opened to the same source can
    make use of this parameter to increase deserialization speed
    - Files on the filesystem and HTTP(S) & S3 streams from
      `stream_io.open_stream` are eligible to be reopened this way
  - The default number of readers is dynamic based on the type of file used
    - To disable concurrent readers, pass `num_readers=1` as a parameter
- Structured object serialization ([#115])
  - `TensorSerializer.write_state_dict` can now write nested mappings,
    sequences, and other mixtures of mappings and sequences nested in each other
  - When accessing an object serialized this way with a `TensorDeserializer`,
    sequences are converted to mappings with integer keys
  - `TensorDeserializer.tree` allows converting the deserialized objects back
    to a compatible collection type
    - Serialized as a sequence → `collections.abc.Sequence`
    - Serialized as a mapping → `collections.abc.Mapping`
  - For more information, see:
    - The `TensorSerializer.write_state_dict` docstring
    - The `TensorDeserializer.tree` docstring
    - PR [#115]
- Configurable CPU concurrency limit during serialization
  - Controlled by the new `limit_cpu_concurrency` `int` parameter
    to the `TensorSerializer` constructor
- New optional keyword parameters to `stream_io.open_stream`:
  - Object storage connection settings `s3_region_name` & `s3_signature_version`
  - File byte range markers `start` & `end`
    - `start` applies to all files and streams
    - `end` applies only to HTTP(S) & S3 streams, for which it is interpreted
      as the `start` and `end` parameters for the created `CURLStreamFile`
      object

[#87]: https://github.com/coreweave/tensorizer/pull/87
[#115]: https://github.com/coreweave/tensorizer/pull/115

### Changed

- The `plaid_mode` and `plaid_mode_buffers` parameters to `TensorDeserializer`
  no longer have an effect
  - The previous default behaviour (`plaid_mode=True` wherever available)
    is now always applied
- Serialization performance has been improved
- `TensorDeserializer.read_tensors` now returns tensors on the target device,
  and functions more efficiently
  - Previously, the returned values were always on the CPU
- `TensorDeserializer.read_tensors`'s behaviour is no longer affected by
  the position of the file descriptor at the time of the call
  - Sequential calls to `read_tensors` still read consecutive parts of the file
- Importing `tensorizer` doesn't implicitly initialize `torch.cuda`
  whenever a GPU is available
  - This allows forking after importing `tensorizer`, and using the library
    in a subprocess
- `TensorDeserializer.read_numpy_arrays` now throws an error when used with
  CUDA deserialization, since numpy arrays can't be deserialized to CUDA

### Fixed

- Fixed a bug where `stream_io.CURLStreamFile` objects constructed with an
  `end` parameter would read one byte past their end when calling
  `CURLStreamFile.read` with no argument

## [2.8.1] - 2024-02-15

### Changed

- Performance has been improved when serializing to some filesystems
  (e.g. NFS, CephFS) by skipping `fallocate` pre-allocation where it
  is not natively supported
  - Previously, `posix_fallocate`'s fallback behaviour was used, which
    wasted time writing out zeroes that would only be overwritten later

### Fixed

- `examples/hf_serialization.py` is now more robust when overwriting an existing
  serialized model in an object storage bucket
  - Previously, it would sometimes find and use outdated, cached data,
    and thus erroneously skip serialization and/or fail validation

## [2.8.0] - 2024-02-08

### Added

- Tensors on the `meta` device may now be serialized
  - These store no tensor data (only metadata) in the tensorized file
  - These have no hashes for their tensor data, since there is nothing to hash
  - These cannot have their data encrypted, since there is nothing to encrypt
  - During deserialization, these are returned as zero-filled buffers on
    the same device as other tensors
    - Essentially equivalent to `torch.zeros_like(meta_tensor, device=...)`

### Changed

- `TensorDeserializer` now defaults to `plaid_mode=True`
  when deserializing to CUDA devices for better performance
  - There is no difference between `plaid_mode`-deserialized tensors
    and regular deserialized tensors (beyond deserialization performance),
    so this is not a breaking change
- Removed incorrect warnings in the documentation about `plaid_mode`
  being unsafe

### Fixed

- Passing `include_non_persistent_buffers=False` to
  `TensorSerializer.write_module()` now works as intended
  - Previously, setting this flag to `False` filtered out both non-persistent
    buffers **and** parameters, leaving only persistent buffers
  - The corrected behaviour only filters out non-persistent buffers,
    leaving parameters untouched
- Very large individual tensors (over approximately 2147479552 bytes)
  now serialize correctly
  - Previously, anything over the limit for a single `write` or `pwrite` syscall
    could not be fully written, and an error was raised during serialization
  - Now, multiple writes are used
  - This also fixes large writes to unbuffered file-like objects if `pwrite`
    is not supported, as they would encounter the same issue

## [2.7.2] - 2024-01-30

### Fixed

- File objects opened with `stream_io.open_stream("s3://...", "wb")` for writing
  to object storage now correctly upload their content when closed implicitly
  at the end of a `with` block, without requiring an explicit call to their
  `.close()` method
  - Since `TensorSerializer` objects already call `.close()` explicitly on
    their output file objects, either when `TensorSerializer.close()` is invoked
    or when the `TensorSerializer` is garbage collected, this bug mainly applies
    to manual usage of `stream_io.open_stream()` for object storage uploads
    not involving a `TensorSerializer`

## [2.7.1] - 2023-12-06

### Fixed

- Fixed a bug where a `CURLStreamFile` would report itself as unreadable,
  causing HTTP(S) and S3 deserialization to fail

## [2.7.0] - 2023-12-06

### Added

- Tensor encryption
  - Refer to [docs/encryption.md](/docs/encryption.md) for details 
  - Encrypts all tensor weights in a file with minimal overhead
  - Doesn't encrypt tensor metadata, such as:
    - Tensor name
    - Tensor `dtype`
    - Tensor shape & size
  - Requires an up-to-date version of `libsodium`
    - Use `apt-get install libsodium23` on Ubuntu or Debian
    - On other platforms, follow the
      [installation instructions from the libsodium documentation](https://doc.libsodium.org/installation)
    - Takes up less than 500 KiB once installed
  - Uses a parallelized version of XSalsa20-Poly1305 as its encryption algorithm
    - Splits each tensor's weights into &leq; 2 MiB chunks, encrypted separately
  - Example usage: see [examples/encryption.py](examples/encryption.py)
  - Example CLI tool to add or remove encryption from pre-serialized models:
    [examples/encrypt_existing.py](examples/encrypt_existing.py)

### Changed

- Added more error checking against deserializing corrupted files
- Added stricter error checking for file writes during serialization

### Fixed

- Fix cases where the `pynvml` library was available on a node with no NVML
  devices
  - This allows CPU-only deployments to work with `pynvml` in the image
- Fix serialization for tensors with discontiguous memory
- Fixed a bug where the `module_idx` on bulk serialized tensors was misaligned
  - During bulk writes (`write_module()`, `write_state_dict()`),
    each tensor was receiving the preceding one's `module_idx`
    instead of its own

## [2.6.0] - 2023-10-30

### Added

- `TensorSerializer.write_module` now accepts `include_non_persistent_buffers`
  as a keyword-only boolean argument that can be set to `False` to exclude
  buffers from serialization that were originally registered to the module
  through calling `torch.nn.Module.register_buffer` with `persistent=False`
  - `torch.nn.Module.state_dict` never includes persistent buffers,
    so setting this to `False` will more closely match the behaviour
    of `state_dict` serialization
  - `TensorSerializer.write_module` used to always include non-persistent
    buffers
  - The default (`include_non_persistent_buffers=True`) matches the old
    behaviour
- `stream_io.open_stream` and `stream_io.CURLStreamFile` now accept an
  additional, optional `certificate_handling` argument to customize
  the verification of SSL certificates
  - This corresponds to the flags
    [`--cacert`](https://curl.se/docs/manpage.html#--cacert),
    [`--capath`](https://curl.se/docs/manpage.html#--capath), and
    [`-k`/`--insecure`](https://curl.se/docs/manpage.html#-k) in `curl`
  - Customization is achieved by passing an instance of `stream_io.CAInfo`
    to `open_stream` or the `CURLStreamFile` constructor
  - Example usages:
    - `open_stream("https://localhost/model.tensors", certificate_handling=CAInfo(cacert="./localhost.pem")`
    - `open_stream("https://127.0.0.1/model.tensors", certificate_handling=CAInfo(allow_untrusted=True)`
  - Pass `certificate_handling=None` (the default) to use default certificate
    verification as compiled into cURL

## [2.5.1] - 2023-10-17

### Changed

- `TensorSerializer.write_state_dict` has been optimized to better match the
  speed of `TensorSerializer.write_module`
- Improved error tracebacks reported during bulk tensor deserialization

### Fixed

- Serializing to a buffered file-like object with a large buffer size
  no longer sometimes corrupts the resulting serialized file

## [2.5.0] - 2023-10-13

### Added

- `TensorDeserializer` now takes a `plaid_mode_buffers` argument specifying
  a fixed number of buffers to allocate when `plaid_mode=True`
  - Previously, `plaid_mode` used a single buffer
  - More buffers help when loading from very fast sources
    or when `verify_hash=True`
  - The new default number of buffers is contextual
    - 1 for HTTP/S3 streams
    - 2 for other streams (e.g. local files, Redis)
    - 8 when `verify_hash=True`
- `TensorDeserializer` objects can now be used as context managers to safely
  call `TensorDeserializer.close` when they are done being used

### Changed

- `TensorDeserializer` methods that load multiple tensors at a time
  are now faster
- `TensorDeserializer`'s `verify_hash` mode is much, much faster
- Specifying `plaid_mode=True` for a `TensorDeserializer` no longer implies
  (or requires) `lazy_load=True`
  - The old default behaviour can be restored by specifying both
    `plaid_mode=True, lazy_load=True`
- `plaid_mode` no longer prohibits accessing previously loaded tensors
- `dtype` conversion is more efficient for CUDA tensor deserialization
  - Conversions are now performed on-device rather than on the CPU
- CPU memory is now freed immediately after `TensorDeserializer` initialization
  for CUDA tensor deserialization when `lazy_load=False`

### Fixed

- `TensorDeserializer`'s `lazy_load` mode no longer eagerly allocates
  memory that is never used

## [2.4.0] - 2023-10-05

### Added

- Support for `redis://` URIs in `stream_io.open_stream`
  - E.g. `redis://localhost:6379/mymodel`
- New `stream_io.RedisStreamFile` class
  - Similar to `stream_io.CURLStreamFile`
- `TensorDeserializer.to_redis` method for initially loading tensors into
  a Redis data store
- `force_http` parameter to `stream_io.open_stream` to downgrade an S3
  connection from HTTPS to HTTP
  - **Warning!** This will stream all data completely unencrypted
  - **Warning!** If accessing a private S3 bucket, this will also send
    your object-scoped access key to the server unencrypted
- `buffer_size` parameter to `stream_io.open_stream` to control the amount of
  data buffered in advance during HTTP(S) loading
  - Defaults to 16 MiB for HTTP(S) streams and 1 to 8 MiB for Redis streams
  - Previously, this was fixed at 256 MiB

### Changed

- `TensorSerializer.write_module` has been optimized further for a speedup of
  ~3.6x on CUDA modules and ~3.1x on CPU modules
- `redis` and `hiredis` are now required package dependencies

### Fixed

- `CURLStreamFile.response_headers` no longer has a chance to contain incomplete
  header information

## [2.3.0] - 2023-09-06

### Added

- `CURLStreamFile` now tracks request headers in
  `CURLStreamFile.response_headers`
  - This can be used to track cache hits and misses during deserialization
    through the `TensorDeserializer.cache_status` property

## [2.2.0] - 2023-09-05

### Changed

- Model serialization has been optimized for a speedup of approximately ~2x

## [2.1.2] - 2023-08-17

### Changed

- Requests now include a custom `User-Agent` header specific to `tensorizer`

## [2.1.1] - 2023-08-10

### Added

- `verify_hash` parameter for `TensorDeserializer.read_tensors`
  - Matches the one for `TensorDeserializer.read_numpy_arrays`

## [2.1.0] - 2023-08-09

### Added

- Hash verification of deserialized models
  - During deserialization, specify `verify_hash=True` in either:
    - The `TensorDeserializer` constructor,
    - `TensorDeserializer.read_numpy_arrays`, or
    - `TensorDeserializer.load_into_module` (only while lazy loading)
  - Comparing a model already in memory against its `.tensors` file:
    `TensorDeserializer.verify_module`

## [2.0.0] - 2023-06-07

### Added

- `bfloat16` and `complex32` support

### Changed

- Newly serialized files now use the `TENSORIZER_VERSION = 2` binary format
  - Format v2 allows for `bfloat16` and `complex32` dtypes to be stored
  - Existing format v1 files can still be deserialized (backwards-compatible)
- `TensorDeserializer`'s `dtype` parameter now only accepts the types
  `torch.dtype` and `None`
  - It previously accepted `numpy.dtype`, `str`, and `None`
- `TensorDeserializer.read_tensors` now yields `torch.Tensor` objects instead of
  `numpy.ndarray` objects
  - `TensorDeserializer.read_numpy_arrays` provides the old functionality
    - Will error when deserializing `bfloat16` or `complex32` by default, since
      they are not valid dtypes in `numpy`
    - The parameter `allow_raw_data` can be specified to read `bfloat16` and
      `complex32` arrays anyway *but with an invalid dtype*

### Fixed

- `TensorDeserializer`'s `plaid_mode` now correctly implies `lazy_load`

## [1.1.0] - 2023-05-05

### Added

- Better docstrings for the public `tensorizer` interface
- More memory utilities in `utils`:
  - `MemoryUsage`: Same information as `get_mem_usage` as a structured type
  - `GlobalGPUMemoryUsage`: GPU information subset of `MemoryUsage`
  - `TorchGPUMemoryUsage`: Torch information subset of `MemoryUsage`
  - `CPUMemoryUsage`: CPU information subset of `MemoryUsage`

### Changed

- `utils.no_init_or_tensor` can now be used as a context manager

## [1.0.1] - 2023-03-21

### Changed

- Loading from public-read S3 buckets no longer requires blank credentials
  to be explicitly specified via `stream_io.open_stream`

## [1.0.0] - 2023-03-21

### Added

- `TensorSerializer` class
- `TensorDeserializer` class
- State dict compatibility
- File, HTTP(S), and S3 stream compatibility
- `stream_io` module and `stream_io.open_stream` interface
- `s3://tensorized` public bucket hosting pre-serialized models
- `utils` module including:
  - `convert_bytes`
  - `get_device`
  - `get_mem_usage`
  - `get_gpu_name`
  - `no_init_or_tensor`

[Unreleased]: https://github.com/coreweave/tensorizer/compare/v2.11.1...HEAD
[2.11.1]: https://github.com/coreweave/tensorizer/compare/v2.11.0...v2.11.1
[2.11.0]: https://github.com/coreweave/tensorizer/compare/v2.10.1...v2.11.0
[2.10.1]: https://github.com/coreweave/tensorizer/compare/v2.10.0...v2.10.1
[2.10.0]: https://github.com/coreweave/tensorizer/compare/v2.9.3...v2.10.0
[2.9.3]: https://github.com/coreweave/tensorizer/compare/v2.9.2...v2.9.3
[2.9.2]: https://github.com/coreweave/tensorizer/compare/v2.9.1...v2.9.2
[2.9.1]: https://github.com/coreweave/tensorizer/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/coreweave/tensorizer/compare/v2.8.1...v2.9.0
[2.8.1]: https://github.com/coreweave/tensorizer/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/coreweave/tensorizer/compare/v2.7.2...v2.8.0
[2.7.2]: https://github.com/coreweave/tensorizer/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/coreweave/tensorizer/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/coreweave/tensorizer/compare/v2.6.0...v2.7.0
[2.6.0]: https://github.com/coreweave/tensorizer/compare/v2.5.1...v2.6.0
[2.5.1]: https://github.com/coreweave/tensorizer/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/coreweave/tensorizer/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/coreweave/tensorizer/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/coreweave/tensorizer/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/coreweave/tensorizer/compare/v2.1.2...v2.2.0
[2.1.2]: https://github.com/coreweave/tensorizer/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/coreweave/tensorizer/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/coreweave/tensorizer/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/coreweave/tensorizer/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/coreweave/tensorizer/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/coreweave/tensorizer/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/coreweave/tensorizer/releases/tag/v1.0.0
