import dataclasses
import json
import operator
import types
import typing
from typing import Callable, Dict, Iterable, Iterator, List, Tuple, Union

# Tensor paths are made up of strings (for mapping keys)
# and integers (for array indices)
_TensorPathComponent: "typing.TypeAlias" = Union[str, int]


class _TensorPath(tuple):
    def serialized_(self) -> bytes:
        if self.is_str_:
            return self[0].encode("utf-8")
        else:
            # application/json-seq format
            return b"\x1e" + json.dumps(
                self, indent=None, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")

    @property
    def is_str_(self) -> bool:
        return len(self) == 1 and isinstance(self[0], str)

    def normalize_(self) -> Union[tuple, str]:
        return self[0] if self.is_str_ else tuple(self)

    def __str__(self) -> str:
        return str(self.normalize_())

    def append_(self, other: Union[str, int]) -> "_TensorPath":
        if not isinstance(other, (str, int)):
            raise TypeError(f"Invalid key type: {other.__class__.__name__!r}")
        else:
            return self.__class__(self + (other,))

    def validate_(self) -> None:
        if not self:
            raise ValueError("Invalid empty tensor path")
        for i in self:
            if not isinstance(i, (str, int)):
                raise TypeError(
                    "Invalid tensor path component type:"
                    f" {i.__class__.__name__!r}"
                )
            if isinstance(i, int) and i < 0:
                raise ValueError(
                    f"Invalid negative integer tensor path component: {i}"
                )

    @classmethod
    def wrap_(cls, value: Union["_TensorPath", tuple, str]) -> "_TensorPath":
        if isinstance(value, cls):
            return value
        elif isinstance(value, tuple):
            return cls(value)
        else:
            return cls((value,))

    @staticmethod
    def _invalid_hook(*_args, **_kwargs):
        raise TypeError("Invalid deserialized type")

    @classmethod
    def deserialize_(cls, serialized: typing.ByteString) -> "_TensorPath":
        if not isinstance(serialized, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Invalid tensor path: expected byte string,"
                f" got {serialized.__class__.__name__!r}"
            )
        if not serialized:
            ret = cls()
            ret.validate_()
            return ret
        is_mv: bool = isinstance(serialized, memoryview)
        first_byte: int = serialized[0]
        if first_byte == 0x1E:
            if (
                len(serialized) < 3
                or serialized[1] != 0x5B  # "["
                or serialized[-1] != 0x5D  # "]"
            ):
                # Require the form <RS>[...]
                raise ValueError("Invalid tensor path: non-array json-seq")
            if 0x0A in serialized or 0x0D in serialized:
                raise ValueError("Illegal newline in json-seq")
            try:
                deserialized: List[Union[str, int]] = json.loads(
                    serialized[1:].tobytes() if is_mv else serialized[1:],
                    object_hook=cls._invalid_hook,
                    parse_float=cls._invalid_hook,
                    parse_constant=cls._invalid_hook,
                )
            except RecursionError as e:
                raise ValueError(
                    "Cannot deserialize tensor path due to excessive nesting"
                ) from e
            if not isinstance(deserialized, list):
                raise TypeError(
                    "Invalid deserialized type:"
                    " expected array as top level object"
                )
            ret = cls(deserialized)
            ret.validate_()
            return ret
        else:
            if is_mv:
                serialized = serialized.tobytes()
            return cls((serialized.decode("utf-8"),))


@dataclasses.dataclass
class _TensorPathRegistry:
    """
    Tracks tensor paths used so far, throwing an error on prefix conflicts,
    and building a prefix tree of layers of the nested structure.
    """

    __slots__ = "_registered_paths"
    _registered_paths: dict

    def __init__(self):
        self._registered_paths = {}

    def _check_compatible_types(self, path: _TensorPath) -> None:
        branch = self._registered_paths
        for depth, component in enumerate(path):
            if not branch:
                break
            existing_type = type(next(iter(branch)))
            current_type = type(component)
            if existing_type is not current_type:
                prefix: tuple = path[: depth + 1]
                raise ValueError(
                    "Conflicting tensor paths:"
                    f" {path.normalize_()} has a different key type"
                    f" ({current_type.__name__!r}) than existing keys at the"
                    f" prefix {prefix} ({existing_type.__name__!r})"
                )
            if component not in branch:
                break
            branch = branch[component]

    def register_path(self, path: Union[_TensorPath, str]) -> None:
        branch: dict = self._registered_paths
        if isinstance(path, str):
            path = _TensorPath((path,))
        if not isinstance(path, _TensorPath):
            raise TypeError(
                f"Invalid tensor path type: {path.__class__.__name__!r}"
            )
        if not path:
            raise ValueError("Invalid empty tensor path")
        self._check_compatible_types(path)
        for component in path[:-1]:
            branch = branch.setdefault(component, {})
            if not isinstance(branch, dict):
                raise ValueError(f"Conflicting tensor paths: {path}, {branch}")
        component = path[-1]
        if component in branch:
            if isinstance(branch[component], dict):
                raise ValueError(
                    f"Conflicting tensor paths: {path.normalize_()} is both"
                    " a leaf and a prefix of another path"
                )
            else:
                raise ValueError(
                    "Conflicting tensor paths:"
                    f" {path.normalize_()} is used multiple times"
                )
        branch[component] = path

    def filter(self, leaf_filter: Callable[[_TensorPath], bool]):
        layers = [(self._registered_paths, iter(tuple(self._registered_paths)))]
        while layers:
            layer, layer_keys = layers[-1]
            for k in layer_keys:
                v = layer[k]
                if isinstance(v, _TensorPath):
                    # If this is a leaf, check if it needs to be pruned
                    if not leaf_filter(v):
                        del layer[k]
                else:
                    # Otherwise, recurse
                    layers.append((v, iter(tuple(v))))
                    break
            else:
                layers.pop()

    def dict(self) -> dict:
        return self._registered_paths


def key_value_iterator(obj: Union[typing.Sequence, typing.Mapping]):
    if isinstance(obj, typing.Mapping):
        for k in obj.keys():
            if not isinstance(k, str):
                raise TypeError(
                    "Invalid key type for state_dict: expected str, got"
                    f" {k.__class__.__name__!r}"
                )
        return iter(obj.items())
    elif isinstance(obj, typing.Sequence):
        return enumerate(obj)
    else:
        raise TypeError(
            "Cannot serialize type as part of a state_dict:"
            f" {obj.__class__.__name__!r}"
        )


_LeafType = typing.TypeVar("_LeafType")


def flatten_structure(
    leaf_type: typing.Type[_LeafType],
    obj: Union[List, typing.Mapping],
    prefix: _TensorPath = _TensorPath(),
) -> Iterable[Tuple[_TensorPath, _LeafType]]:
    iters: List[Tuple[_TensorPath, Iterator]] = [
        (prefix, key_value_iterator(obj))
    ]
    while iters:
        pre, it = iters[-1]
        for name, item in it:
            path: _TensorPath = pre.append_(name)
            if isinstance(item, leaf_type):
                yield path, item
            else:
                iters.append((path, key_value_iterator(item)))
                break
        else:
            iters.pop()


def restructure(
    flat: Dict[_TensorPath, _LeafType], use_dict_proxies: bool = False
) -> Union[dict, list, types.MappingProxyType]:
    for path in flat.keys():
        if len(path) < 1:
            raise ValueError("Invalid empty tensor path key")

    # Start reconstructing everything as nested dictionaries
    base = {}
    for path, tensor in flat.items():
        branch = base
        for component in path[:-1]:
            branch = branch.setdefault(component, {})
            if not isinstance(branch, dict):
                # Key path conflicts should be caught at the metadata
                # parsing step, so this is just an extra sanity check
                raise RuntimeError(f"Key path conflict for key {path}")
        component = path[-1]
        if component in branch:
            raise RuntimeError(f"Key path conflict for key {path}")
        branch[component] = tensor

    # Assign a type to each layer separately
    def re_type_layer(
        untyped_layer: dict,
    ) -> Union[dict, list, types.MappingProxyType]:
        if use_dict_proxies:
            return types.MappingProxyType(untyped_layer)
        is_list = False
        for key in untyped_layer:
            if isinstance(key, int):
                is_list = True
                if key < 0:
                    raise ValueError(
                        "Illegal negative integer tensor path component"
                    )
            elif is_list:
                raise ValueError(
                    "Invalid tensor path keys:"
                    " mixes dict and list on same layer"
                )
        if is_list:
            # Lists are always ordered by the value of their key indices,
            # rather than the order in the file.
            list_layer = list(untyped_layer.items())
            list_layer.sort(key=operator.itemgetter(0))
            return [v for _, v in list_layer]
        else:
            return untyped_layer

    # Track recursive state with a stack.
    # Iterators track progress through the keys of each layer.
    # Direct iterators over dictionaries (rather than just keys)
    # may not be stable while actively mutating the dictionary
    # mid-iteration, so the iterators are over a stable copy
    # of the keys instead.
    base_iter = iter(tuple(base))
    layers = [(None, None, base, base_iter)]
    while layers:
        last_layer, last_key, layer, key_iterator = layers[-1]
        for k in key_iterator:
            next_layer = layer[k]
            if isinstance(next_layer, dict):
                # Recurse
                next_iter = iter(tuple(next_layer))
                layers.append((layer, k, next_layer, next_iter))
                break
        else:
            # Update the key in the parent (or base) with the corrected type
            re_typed = re_type_layer(layer)
            if last_layer is None:
                base = re_typed
            else:
                last_layer[last_key] = re_typed
            layers.pop()
    return base
