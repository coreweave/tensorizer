---
title: Tensor Encryption
---
`tensorizer` supports fast tensor weight encryption and decryption during
serialization and deserialization, respectively.

> [!NOTE]
>
> To use `tensorizer` encryption, a recent version of `libsodium` must be
> installed. Install `libsodium` with `apt-get install libsodium23`
> on Ubuntu or Debian, or follow
> [the instructions in `libsodium`'s documentation](https://doc.libsodium.org/installation)
> for other platforms.

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

### Encrypting and Decrypting Existing Models with a CLI Tool

Existing models that have already been serialized by tensorizer can have
encryption added to them or removed from them using the example
[`examples/encrypt_existing.py`](/examples/encrypt_existing.py)
command line utility. Download the script, then run
`python encrypt_existing.py -h` to see usage.

The source code in [`examples/encrypt_existing.py`](/examples/encrypt_existing.py)
also serves as usage examples for the various encryption methods.

Examples:

```bash
# Global help and subcommand help
python encrypt_existing.py --help
python encrypt_existing.py add pwhash --help

# Encrypt using a random binary key (outputs generated key to --keyfile)
python encrypt_existing.py add random --keyfile encrypted.tensors.key \
  --infile original.tensors --outfile encrypted.tensors

# Encrypt using a pre-existing binary key (reads key from --keyfile)
python encrypt_existing.py add exact --keyfile encrypted.tensors.key \
  --infile original.tensors --outfile encrypted.tensors

# Encrypt using Argon2id key derivation (reads string to turn into a key from --keyfile)
python encrypt_existing.py add pwhash --keyfile encrypted.tensors.key \
  --opslimit MODERATE --memlimit MODERATE \
  --infile original.tensors --outfile encrypted.tensors

# Decrypt using a binary key (reads key from --keyfile)
python encrypt_existing.py remove exact --keyfile encrypted.tensors.key \
  --infile encrypted.tensors --outfile decrypted.tensors

# Decrypt using Argon2id key derivation (reads string to turn into a key from --keyfile)
python encrypt_existing.py remove pwhash --keyfile encrypted.tensors.key \
  --infile encrypted.tensors --outfile decrypted.tensors
```

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

### Using `EncryptionParams.from_string()` with an environment variable

If an encryption key must be provided as a pre-existing string,
this method of encryption will allow the use of that string,
and does not require saving a key generated at the time of encryption.

> [!WARNING]
>
> Make sure a secure input string is used to create a key.
> `EncryptionParams.from_string()` accepts parameters to tune its algorithm
> to make searching the input string via brute-force checking less viable,
> but nothing can protect against weak enough input strings,
> like your birthdate, or a common password.

```py
from tensorizer import (
    EncryptionParams, DecryptionParams, TensorDeserializer, TensorSerializer
)

source: str = os.getenv("SUPER_SECRET_STRONG_PASSWORD")

# Serialize and encrypt a model:
encryption_params = EncryptionParams.from_string(source)
serializer = TensorSerializer("model.tensors", encryption=encryption_params)
serializer.write_module(...)  # or write_state_dict(), etc.
serializer.close()

# Then decrypt it again:
decryption_params = DecryptionParams.from_string(source)
deserializer = TensorDeserializer("model.tensors", encryption=decryption_params)
deserializer.load_into_module(...)
deserializer.close()
```

### Choosing a Key Derivation Algorithm

The classes `EncryptionParams` and `DecryptionParams` allow a choice of
key derivation method. Two methods are implemented:

1. Random key generation
    1. Chosen by constructing an `EncryptionParams` object through calling
       `EncryptionParams.random()`
    2. Not technically key derivation
    3. Uses a completely random 32-byte sequence with no associated passphrase
    4. Highly secure against being guessed
    5. You must save the randomly generated key
2. [Argon2id](https://datatracker.ietf.org/doc/html/rfc9106) key derivation
    1. Chosen by constructing an `EncryptionParams` object through calling
       `EncryptionParams.from_string(source)`
    2. Transmutes an arbitrary-length `str` or `bytes` source string into a
       binary encryption key
    3. Implements adjustable security against brute-force cracking
       via its `opslimit` and `memlimit` parameters
    4. Internally uses
       [`libsodium`'s `pwhash` function with the algorithm `crypto_pwhash_ALG_ARGON2ID13`](https://libsodium.gitbook.io/doc/password_hashing/default_phf#key-derivation)

An `EncryptionParams` object is passed to a `TensorSerializer` using its
`encryption=...` keyword-only parameter during initialization.

#### `EncryptionParams.from_string()` details (Argon2id)

`EncryptionParams.from_string()` uses the Argon2 (Argon2id, RFC 9106)
password hashing algorithm to create a key from an input string.

The key has resistance against brute-force attacks that attempt
to guess the input string, achieved by making each attempt
expensive to compute, both in CPU time and RAM usage.

The computational difficulty can be increased or decreased
via the `opslimit` and `memlimit` parameters.
Higher computational difficulty gives more security
for weak input strings, but may impact performance.
The default setting is a "moderate" profile taken from `libsodium`.

Presets (as well as minimum values) are available through the
`EncryptionParams.OpsLimit` and `EncryptionParams.MemLimit` enums.

Rough estimates of performance impact (on a 3.20 GHz processor):

```py
from tensorizer import EncryptionParams

OpsLimit = EncryptionParams.OpsLimit
MemLimit = EncryptionParams.MemLimit
s = "X" * 40

EncryptionParams.from_string(  # Takes about 0.05 ms, 8 KiB RAM
    s, opslimit=OpsLimit.MIN, memlimit=MemLimit.MIN
)
EncryptionParams.from_string(  # Takes about 90 ms, 64 MiB RAM
    s, opslimit=OpsLimit.INTERACTIVE, memlimit=MemLimit.INTERACTIVE
)
EncryptionParams.from_string(  # Takes about 500 ms, 256 MiB RAM
    s, opslimit=OpsLimit.MODERATE, memlimit=MemLimit.MODERATE
    # Default: equivalent to opslimit=None, memlimit=None
)
EncryptionParams.from_string(  # Takes about 3.0 seconds, 1 GiB RAM
    s, opslimit=OpsLimit.SENSITIVE, memlimit=MemLimit.SENSITIVE
)
```

Timing may be different on different hardware.
These do not reflect the exact times an attacker may require for each guess.

##### Performance tuning

If possible, use `EncryptionParams.random()` instead of
`EncryptionParams.from_string()`, and save the generated key
to use for decryption.

If that is not possible, save the binary key generated during
`EncryptionParams.from_string()` (from the `.key` attribute),
and use that key for decryption (via `DecryptionParams.from_key()`)
to remove the cost of re-computing the key at deserialization time.

If that is not possible, use a strong input string.
For input strings that are already very strong and high-entropy,
where brute-force attacks on the input string are no more likely
to succeed than brute-force attacks on a 256-bit key itself,
(e.g. very long, randomly generated strings),
`opslimit` and `memlimit` may be tuned down to minimize
their performance impact.

If that is not possible, test different values of `opslimit`
and `memlimit` to determine an acceptable tradeoff between
performance and security for your use case.

See also:
- [`libsodium` documentation for `pwhash`](https://libsodium.gitbook.io/doc/password_hashing/default_phf#key-derivation),
  the Argon2id implementation used in `EncryptionParams.from_string()`
- [RFC 9106](https://datatracker.ietf.org/doc/html/rfc9106)
  for details on Argon2 and Argon2id

#### Using the right key derivation algorithm for decryption

Specifying whether to decrypt with a passphrase or key is done with
a `DecryptionParams` object.
A `DecryptionParams` object is passed to a `TensorDeserializer` using its
`encryption=...` keyword-only parameter during initialization.

When passphrase-based key derivation is used during encryption,
*key derivation metadata* recording the algorithm used is stored
in the tensorized file. Since the file keeps track of the algorithm,
any `from_string()`-based encryption can be decrypted the same way:

```py
source: str = ...
decryption_params = DecryptionParams.from_string(source)
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
encryption_params = EncryptionParams.from_string(...)

key: bytes = encryption_params.key
```

## Speed

The throughput of `tensorizer`'s encryption algorithm reaches 31 GiB/s on
~26 cores, most likely limited by RAM or CPU cache speed.
Since it can overlap with downloads, the time overhead of decryption is
very small, with data-processed-to-latency-incurred rates in the terabit range
encountered on test machines.

Speed of key derivation is configurable, if used.
See [Choosing a Key Derivation Algorithm](#choosing-a-key-derivation-algorithm).

## Compatibility

Tensors serialized with encryption are stored using Tensorizer data format v3,
compatible to be read with `tensorizer>=2.7.0`.
