# Tensor Encryption

`tensorizer` supports fast tensor weight encryption and decryption during
serialization and deserialization, respectively.

## Encryption Algorithm

Tensor encryption splits weights into up-to-2 MiB chunks encrypted independently
with XSalsa20-Poly1305 symmetric authenticated encryption,
stored separately from their MACs.
These chunks can be encrypted or decrypted and authenticated independently of
one another in the style of a block cipher,
which allows decryption parallelized with streaming.

All encryption and decryption is done in-place, as encrypted payloads are equal
in length to their plaintexts in this scheme (since the MACs are stored
separately).
This allows for high speed processing, since memory allocations can be avoided.

## Scope / Security

Only tensor weights are encrypted, using 256-bit keys (see
[Choosing a Key Derivation Algorithm](#choosing-a-key-derivation-algorithm)),
and weight chunks are independently authenticated.
This is meant to provide security against tensor weights from being read by a
third party, plus a small amount of authentication to confirm that,
for example, a matching passphrase was used for encryption and decryption;
security beyond that is beyond the scope of `tensorizer`'s encryption as
currently available.

> [!WARNING]
> 
> This does not include encryption for anything except for tensor weights.
> Metadata such as a tensor's name, dtype, shape, size, and non-keyed hashes
> are neither encrypted nor authenticated.

> [!WARNING]
> 
> This level of encryption does not provide message authentication for metadata
> and does not protect against reordering or truncation of chunks.

> [!NOTE]
> 
> Unencrypted/unauthenticated tensor data is rejected during deserialization
> if decryption is requested, and vice versa.

## Usage

A full usage example is given in
[`examples/encryption.py`](/examples/encryption.py).

The class docstrings of `EncryptionParams` and `DecryptionParams` include
usage outlines like below as well as additional usage information.
Most IDEs support automatically displaying this information while coding.

### Using `EncryptionParams.random()`

This is the preferred method of tensor encryption and decryption.
Use this unless you have a good reason to do otherwise.

```py
from tensorizer import (
    EncryptionParams, DecryptionParams, TensorDeserializer, TensorSerializer
)

# Serialize and encrypt a model:
encryption_params = EncryptionParams.random()

serializer = TensorSerializer("model.tensors", encryption=encryption_params)
serializer.write_module(...)  # or write_state_dict(), etc.
serializer.close()

# Save the randomly-generated encryption key somewhere
with open("tensor.key", "wb") as key_file:
    key_file.write(encryption_params.key)


# Then decrypt it again:

# Load the randomly-generated key from where it was saved
with open("tensor.key", "rb") as key_file:
    key: bytes = key_file.read()
 
decryption_params = DecryptionParams.from_key(key)

deserializer = TensorDeserializer("model.tensors", encryption=decryption_params)
deserializer.load_into_module(...)
deserializer.close()
```

### Using `EncryptionParams.from_passphrase_fast()` with an environment variable

If an encryption key must be provided as a pre-existing string that
still has high entropy (~256 bits), this method of encryption will allow the
use of that string, and does not require saving a key generated at the time
of encryption.

> [!WARNING]
> 
> This is not secure for human-written or low-entropy strings, as they can be
> brute-forced guessed. Only use this with passphrase strings that are already
> as secure as an encryption key themselves.

```py
from tensorizer import (
    EncryptionParams, DecryptionParams, TensorDeserializer, TensorSerializer
)

# This passphrase must already be a secure key (high-entropy).
# Short or insecure passphrases may be brute-force guessed.
passphrase: str = os.getenv("SUPER_SECRET_STRONG_PASSWORD")

# Serialize and encrypt a model:
encryption_params = EncryptionParams.from_passphrase_fast(passphrase)
serializer = TensorSerializer("model.tensors", encryption=encryption_params)
serializer.write_module(...)  # or write_state_dict(), etc.
serializer.close()

# Then decrypt it again:
decryption_params = DecryptionParams.from_passphrase(passphrase)
deserializer = TensorDeserializer("model.tensors", encryption=decryption_params)
deserializer.load_into_module(...)
deserializer.close()
```

### Choosing a Key Derivation Algorithm

The classes `EncryptionParams` and `DecryptionParams` allow a choice of
key derivation method. Currently, two methods are implemented:

1. Random key generation
    1. Chosen by constructing an `EncryptionParams` object through calling
       `EncryptionParams.random()`
    2. Uses a completely random 32-byte sequence with no associated passphrase
    3. Highly secure against being guessed
    4. You must save the randomly generated key
2. Fast key derivation
    1. Chosen by constructing an `EncryptionParams` object through calling
       `EncryptionParams.from_passphrase_fast(passphrase)`
    2. Transmutes an arbitrary-length `str` or `bytes` passphrase into a
       binary encryption key
    3. Does not implement any security against brute-force checking
       many passphrases (brute-force checking is prevented by using an
       intentionally slow algorithm, **which this is not**)
    4. Must only be used with passphrases that are already secure and
       high-entropy

An `EncryptionParams` object is passed to a `TensorSerializer` using its
`encryption=...` keyword-only parameter during initialization.

#### Using the right key derivation algorithm for decryption

Specifying whether to decrypt with a passphrase or key is done with
a `DecryptionParams` object.
A `DecryptionParams` object is passed to a `TensorDeserializer` using its
`encryption=...` keyword-only parameter during initialization.

When passphrase-based key derivation is used during encryption,
a *key derivation chunk* recording the algorithm used is stored
in the tensorized file. Since the file keeps track of the algorithm,
any passphrase-based encryption can be decrypted the same way:

```py
passphrase: str = ...
decryption_params = DecryptionParams.from_passphrase(passphrase)
deserializer = TensorDeserializer(..., encryption=decryption_params)
```

Additionally, *any* encryption,
whether a passphrase was used during encryption or not,
can be decrypted if you know its exact binary key:

```py
key: bytes = ...
decryption_params = DecryptionParams.from_key(key)
deserializer = TensorDeserializer(..., encryption=decryption_params)
```

This is the only way to decrypt a file that was encrypted using
`EncryptionParams.random()`, since it has no associated passphrase.

To retrieve a binary key from an `EncryptionParams` object, access its `key`
attribute:

```py
encryption_params = EncryptionParams.random()
# Or
encryption_params = EncryptionParams.from_passphrase_fast(...)

key: bytes = encryption_params.key
```

## Speed

The throughput of `tensorizer`'s encryption algorithm reaches 31 GiB/s on
~26 cores, most likely limited by RAM or CPU cache speed.
Since it can overlap with downloads, the time overhead of decryption is
very small, with data-processed-to-latency-incurred rates in the terabit range
encountered on test machines.

## Compatibility

Tensors serialized with encryption are stored using Tensorizer data format v3,
compatible to be read with `tensorizer>=2.7.0`.
