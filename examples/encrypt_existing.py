import argparse
import binascii
from contextlib import ExitStack


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "add or remove encryption from an already-tensorized file. See"
            " docs/encryption.md for an explanation."
        )
    )

    subparsers = parser.add_subparsers(
        help="whether to add or remove encryption"
    )

    LIMITS_METAVAR = "<SENSITIVE|MODERATE|INTERACTIVE|MIN|int>"

    add_parser = subparsers.add_parser(
        "add", description="add encryption to an already-tensorized file"
    )
    add_subparsers = add_parser.add_subparsers(
        help="key derivation / generation method"
    )

    add_pwhash_parser = add_subparsers.add_parser(
        "pwhash",
        description=(
            "encrypt using a key generated with Argon2id key derivation"
        ),
    )
    add_pwhash_parser.set_defaults(func=add_pwhash)
    add_pwhash_parser.add_argument(
        "--keyfile",
        type=argparse.FileType("rb"),
        required=True,
        help=(
            "file holding data to process into an encryption key using Argon2id"
        ),
    )
    add_pwhash_parser.add_argument(
        "--no-strip-trailing-newlines",
        dest="strip_trailing_newlines",
        action="store_false",
        default=True,
        help="don't strip trailing newlines from the key file",
    )

    add_pwhash_parser.add_argument(
        "--opslimit",
        metavar=LIMITS_METAVAR,
        type=str,
        required=True,
        help=(
            "Argon2id opslimit (CPU time difficulty; param from libsodium's"
            " pwhash function)"
        ),
    )
    add_pwhash_parser.add_argument(
        "--memlimit",
        metavar=LIMITS_METAVAR,
        type=str,
        required=True,
        help=(
            "Argon2id memlimit (RAM difficulty; param from libsodium's pwhash"
            " function)"
        ),
    )
    add_pwhash_parser.add_argument(
        "--salt",
        type=str,
        help=(
            "hex representation of a custom 16-byte cryptographic salt to use"
            " (randomly generated otherwise)"
        ),
    )

    add_exact_key_parser = add_subparsers.add_parser(
        "exact",
        description="encrypt using an exact 32-byte binary key, unmodified",
    )
    add_exact_key_parser.set_defaults(func=add_exact_key)
    add_exact_key_parser.add_argument(
        "--keyfile",
        type=argparse.FileType("rb"),
        required=True,
        help=(
            "file holding exactly 32 bytes of binary data to use verbatim as an"
            " encryption key"
        ),
    )

    add_random_key_parser = add_subparsers.add_parser(
        "random",
        description=(
            "encrypt using a random 32-byte binary key, and write the key to a"
            " file"
        ),
    )
    add_random_key_parser.set_defaults(func=add_random_key)
    add_random_key_parser.add_argument(
        "--keyfile",
        type=argparse.FileType("wb"),
        required=True,
        help=(
            "file to write 32 bytes of randomly-generated binary data used as"
            " an encryption key"
        ),
    )

    remove_parser = subparsers.add_parser(
        "remove",
        description="remove encryption from an already-tensorized file",
    )
    remove_subparsers = remove_parser.add_subparsers(
        help="key derivation / generation method"
    )

    remove_pwhash_parser = remove_subparsers.add_parser(
        "pwhash",
        description=(
            "decrypt using a key generated with Argon2id key derivation"
        ),
    )
    remove_pwhash_parser.set_defaults(func=remove_pwhash)
    remove_pwhash_parser.add_argument(
        "--keyfile",
        type=argparse.FileType("rb"),
        required=True,
        help=(
            "file holding data to process into an encryption key using Argon2id"
        ),
    )
    remove_pwhash_parser.add_argument(
        "--no-strip-trailing-newlines",
        dest="strip_trailing_newlines",
        action="store_false",
        default=True,
        help="don't strip trailing newlines from the key file",
    )

    remove_exact_key_parser = remove_subparsers.add_parser(
        "exact",
        description="decrypt using an exact 32-byte binary key, unmodified",
    )
    remove_exact_key_parser.set_defaults(func=remove_exact_key)
    remove_exact_key_parser.add_argument(
        "--keyfile",
        type=argparse.FileType("rb"),
        required=True,
        help=(
            "file holding exactly 32 bytes of binary data to use verbatim as an"
            " encryption key"
        ),
    )

    for subparser in (
        add_pwhash_parser,
        add_exact_key_parser,
        add_random_key_parser,
        remove_pwhash_parser,
        remove_exact_key_parser,
    ):
        subparser.add_argument(
            "--infile", type=str, required=True, help="source file to convert"
        )
        subparser.add_argument(
            "--outfile",
            type=str,
            required=True,
            help="where to write the resulting converted file",
        )
        subparser.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            default=False,
            help="show less output",
        )

    args = parser.parse_args(argv)

    if args.infile == args.outfile:
        parser.error("--infile and --outfile can't be the same")

    if args.func != add_random_key:
        try:
            args.key = args.keyfile.read()
            args.keyfile.close()
        except OSError:
            parser.error("Provided --keyfile path could not be read")
    else:
        args.key = None

    exact_key_length = 32

    if args.func in (add_exact_key, remove_exact_key):
        if len(args.key) != exact_key_length:
            parser.error(
                "Invalid key length:"
                f" got {len(args.key)} bytes, expected {exact_key_length} bytes"
            )
    elif (
        args.func in (add_pwhash, remove_pwhash)
        and args.strip_trailing_newlines
    ):
        args.key = args.key.rstrip(b"\r\n")

    salt_length = 16

    if args.func == add_pwhash:
        if args.salt is not None:
            if len(args.salt) != salt_length * 2:
                parser.error(
                    f"Invalid --salt length (should be {salt_length} bytes ="
                    f" {salt_length * 2} hex characters)"
                )
            try:
                args.salt = binascii.unhexlify(args.salt)
                assert len(args.salt) == salt_length
            except binascii.Error:
                parser.error("Invalid hexadecimal string provided for --salt")

        limit_options = ("SENSITIVE", "MODERATE", "INTERACTIVE", "MIN")
        args.opslimit = args.opslimit.upper()
        args.memlimit = args.memlimit.upper()
        try:
            int(args.opslimit)
        except ValueError:
            if args.opslimit not in limit_options:
                parser.error(
                    "Invalid --opslimit, expected one of "
                    + ", ".join(limit_options)
                    + ", or an integer"
                )
        try:
            int(args.memlimit)
        except ValueError:
            if args.memlimit not in limit_options:
                parser.error(
                    "Invalid --memlimit, expected one of "
                    + ", ".join(limit_options)
                    + ", or an integer"
                )

    return args


def get_limit(value, enumeration) -> int:
    try:
        return int(value)
    except ValueError:
        value = getattr(enumeration, value, None)
        if value is not None:
            return value
        else:
            raise ValueError(
                f"Unrecognized limit: {value}, available:"
                f" {', '.join(v.name for v in enumeration)}"
            )


def add_pwhash(args: argparse.Namespace):
    from tensorizer import EncryptionParams

    opslimit = get_limit(args.opslimit, EncryptionParams.OpsLimit)
    memlimit = get_limit(args.memlimit, EncryptionParams.MemLimit)
    salt = args.salt
    key: bytes = args.key
    encryption_params = EncryptionParams.from_string(
        source=key, opslimit=opslimit, memlimit=memlimit, salt=salt
    )
    add_encryption(encryption_params, args.infile, args.outfile, not args.quiet)
    print("Salt:", binascii.hexlify(encryption_params.salt).decode("ascii"))


def add_exact_key(args: argparse.Namespace):
    from tensorizer import EncryptionParams

    encryption_params = EncryptionParams(key=args.key)
    add_encryption(encryption_params, args.infile, args.outfile, not args.quiet)


def add_random_key(args: argparse.Namespace):
    from tensorizer import EncryptionParams

    encryption_params = EncryptionParams.random()
    args.keyfile.write(encryption_params.key)
    args.keyfile.close()
    add_encryption(encryption_params, args.infile, args.outfile, not args.quiet)


def add_encryption(
    encryption_params, in_file: str, out_file: str, show_progress: bool = True
):
    from tensorizer import TensorDeserializer, TensorSerializer, TensorType

    with ExitStack() as cleanup:
        serializer = TensorSerializer(out_file, encryption=encryption_params)
        cleanup.callback(serializer.close)
        deserializer = TensorDeserializer(
            in_file, device="cpu", lazy_load=True, verify_hash=True
        )
        cleanup.enter_context(deserializer)
        count: int = len(deserializer.keys())
        i = 1
        for (
            module_idx,
            tensor_type,
            name,
            tensor,
        ) in deserializer.read_tensors():
            if show_progress:
                print(f"({i} / {count}) Encrypting {name}")
                i += 1
            tensor_type = TensorType(tensor_type)
            serializer.write_tensor(module_idx, name, tensor_type, tensor)
            # Release memory
            tensor.set_()
            del tensor


def remove_pwhash(args: argparse.Namespace):
    from tensorizer import DecryptionParams

    decryption_params = DecryptionParams.from_string(args.key)
    remove_encryption(
        decryption_params, args.infile, args.outfile, not args.quiet
    )


def remove_exact_key(args: argparse.Namespace):
    from tensorizer import DecryptionParams

    decryption_params = DecryptionParams.from_key(args.key)
    remove_encryption(
        decryption_params, args.infile, args.outfile, not args.quiet
    )


def remove_encryption(
    decryption_params, in_file: str, out_file: str, show_progress: bool = True
):
    from tensorizer import TensorDeserializer, TensorSerializer, TensorType

    with ExitStack() as cleanup:
        serializer = TensorSerializer(out_file)
        cleanup.callback(serializer.close)
        deserializer = TensorDeserializer(
            in_file,
            device="cpu",
            lazy_load=True,
            verify_hash=True,
            encryption=decryption_params,
        )
        cleanup.enter_context(deserializer)
        count: int = len(deserializer.keys())
        i = 1
        for (
            module_idx,
            tensor_type,
            name,
            tensor,
        ) in deserializer.read_tensors():
            if show_progress:
                print(f"({i} / {count}) Decrypting {name}")
                i += 1
            tensor_type = TensorType(tensor_type)
            serializer.write_tensor(module_idx, name, tensor_type, tensor)
            # Release memory
            tensor.set_()
            del tensor


def main(argv=None):
    args = parse_args(argv)
    args.func(args)
    if not args.quiet:
        print("Done")


if __name__ == "__main__":
    main()
