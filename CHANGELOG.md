# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `CURLStreamFile.response_headers` no longer has a chance to have incomplete
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

[Unreleased]: https://github.com/coreweave/tensorizer/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/coreweave/tensorizer/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/coreweave/tensorizer/compare/v2.1.2...v2.2.0
[2.1.2]: https://github.com/coreweave/tensorizer/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/coreweave/tensorizer/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/coreweave/tensorizer/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/coreweave/tensorizer/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/coreweave/tensorizer/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/coreweave/tensorizer/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/coreweave/tensorizer/releases/tag/v1.0.0