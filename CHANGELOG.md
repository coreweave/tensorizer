# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  call `TensorDeserializer.close()` when they are done being used

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