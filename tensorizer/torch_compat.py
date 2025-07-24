"""
Compatibility layer for using ``torch.save`` and ``torch.load`` with tensorizer
as a backend for the serialization of tensors and tensor storages.

Author: Eta Syra

Example:
    An instance of ``torch.nn.Module`` can be serialized as follows::

        import os
        import torch
        from tensorizer.torch_compat import (
            tensorizer_saving, tensorizer_loading
        )

        module: torch.nn.Module = ...

        with tensorizer_saving():
            torch.save(module, "module.pt")

        assert os.path.exists("module.pt")
        assert os.path.exists("module.pt.tensors")

        with tensorizer_loading(device="cuda", num_readers=4):
            deserialized_module = torch.load("module.pt")

    Both `tensorizer_saving` and `tensorizer_loading` can be passed keyword
    arguments to be forwarded to a `TensorSerializer` and `TensorDeserializer`
    object, respectively. They can also be given a ``file_obj`` argument
    to control where they save the sidecar ``.tensors`` file containing
    tensor data.
"""

import contextlib
import contextvars
import functools
import io
import os
import pickle
import threading
import types
import typing
from typing import Final, Optional, Tuple

import torch

from . import TensorDeserializer, TensorSerializer

__all__ = (
    "tensorizer_saving",
    "tensorizer_loading",
)


_tensorizer_file_obj_type: "typing.TypeAlias" = typing.Union[
    io.BufferedIOBase,
    io.RawIOBase,
    typing.BinaryIO,
    str,
    bytes,
    os.PathLike,
    int,
]

_wrapper_file_obj_type: "typing.TypeAlias" = typing.Union[
    _tensorizer_file_obj_type,
    typing.Callable[[torch.types.FileLike], _tensorizer_file_obj_type],
]

_tensorizer_filename: contextvars.ContextVar[
    Optional[_wrapper_file_obj_type]
] = contextvars.ContextVar("_tensorizer_filename", default=None)

_tensorizer_deserializer_kwargs: contextvars.ContextVar[Optional[dict]] = (
    contextvars.ContextVar("_tensorizer_deserializer_kwargs", default=None)
)

_tensorizer_serializer_kwargs: contextvars.ContextVar[Optional[dict]] = (
    contextvars.ContextVar("_tensorizer_serializer_kwargs", default=None)
)


def _storage_device(
    storage: typing.Union[torch.UntypedStorage, torch.TypedStorage],
) -> torch.device:
    if isinstance(storage, torch.TypedStorage):
        return getattr(storage, "_untyped_storage", storage).device
    else:
        return storage.device


def _has_data(
    storage: typing.Union[torch.UntypedStorage, torch.TypedStorage],
) -> bool:
    maybe_untyped = getattr(storage, "_untyped_storage", storage)
    return maybe_untyped.device.type != "meta" and maybe_untyped.data_ptr() != 0


class _TensorizerPickler(pickle.Pickler):
    __filename: Optional[_tensorizer_file_obj_type]
    __tensors: typing.List[torch.Tensor]
    __tensor_ids: typing.Dict[Tuple[typing.Hashable, ...], int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__filename = _tensorizer_filename.get()
        self.__tensors = []
        self.__tensor_ids = {}

    @staticmethod
    def __tensor_key(tensor: torch.Tensor) -> Tuple[typing.Hashable, ...]:
        return (
            tensor.data_ptr(),
            tensor.dtype,
            tensor.shape,
            tensor.layout,
            tensor.stride(),
        )

    def __register_tensor(self, tensor: torch.Tensor) -> int:
        tensors = self.__tensors
        new: int = len(tensors)
        idx: int = self.__tensor_ids.setdefault(self.__tensor_key(tensor), new)
        if idx is new:
            tensors.append(tensor)
        else:
            tensors[idx] = tensor
        return idx

    def dump(self, obj):
        super().dump(obj)
        if self.__tensors:
            if self.__filename is None:
                self.__tensors.clear()
                self.__tensor_idx = 0
                return
            kwargs = _tensorizer_serializer_kwargs.get()
            if kwargs is None:
                kwargs = {}
            serializer = TensorSerializer(self.__filename, **kwargs)
            try:
                serializer.write_state_dict(self.__tensors)
                serializer.close()
            finally:
                self.__tensors.clear()
                self.__tensor_idx = 0

    @staticmethod
    def __storage_to_tensor(
        storage: typing.Union[torch.UntypedStorage, torch.TypedStorage],
    ) -> torch.Tensor:
        # Convert a storage into an equivalent tensor
        # for compatibility with a TensorSerializer
        if not isinstance(storage, torch.UntypedStorage):
            untyped = getattr(storage, "_untyped_storage", None)
            if untyped is None:
                untyped = storage.untyped()
            storage = untyped
        tensor = torch.empty(
            (0,), dtype=torch.uint8, device=storage.device, requires_grad=False
        )
        tensor.set_(storage)
        return tensor

    def persistent_id(self, obj):
        if (
            self.__filename is not None
            and isinstance(obj, (torch.UntypedStorage, torch.TypedStorage))
            and _has_data(obj)
        ):
            tensor_view = self.__storage_to_tensor(obj)
            dtype = getattr(obj, "dtype", None)
            idx: int = self.__register_tensor(tensor_view)
            return "TensorizerPickler", 0, "storage", idx, dtype
        return None

    @staticmethod
    def __wrap_persistent_id(persistent_id_func: callable):
        @functools.wraps(persistent_id_func)
        def _persistent_id(self, obj):
            super_id = super(self.__class__, self).persistent_id(obj)
            if super_id is not None:
                return super_id
            else:
                return persistent_id_func(self, obj)

        return _persistent_id

    def __setattr__(self, key, value):
        if key == "persistent_id":
            value = self.__wrap_persistent_id(value)
        super().__setattr__(key, value)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "persistent_id" in cls.__dict__:
            cls.persistent_id = cls.__wrap_persistent_id(cls.persistent_id)


class _TensorizerUnpickler(pickle.Unpickler):
    __filename: Optional[_tensorizer_file_obj_type]
    __tensors: list

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__filename = _tensorizer_filename.get()
        self.__tensors = []

    def load(self):
        if self.__filename is not None:
            kwargs = _tensorizer_deserializer_kwargs.get()
            if kwargs is None:
                kwargs = {}
            with TensorDeserializer(self.__filename, **kwargs) as deserializer:
                self.__tensors = deserializer.tree()
        try:
            return super().load()
        finally:
            self.__tensors.clear()

    @staticmethod
    def __tensor_to_storage(
        tensor: torch.Tensor, dtype: Optional[torch.dtype]
    ) -> typing.Union[torch.UntypedStorage, torch.TypedStorage]:
        # Convert a tensor into an equivalent storage
        # for compatibility with a TensorDeserializer
        if dtype is None:
            # Oddly, PyTorch expects a storage serialized as an UntypedStorage
            # to be loaded as a TypedStorage with the torch.uint8 type.
            dtype = torch.uint8
        return torch.TypedStorage(
            wrap_storage=tensor.untyped_storage(), dtype=dtype, _internal=True
        )

    def persistent_load(self, pid):
        if (
            self.__filename is not None
            and isinstance(pid, tuple)
            and pid[0] == "TensorizerPickler"
            and len(pid) >= 3
        ):
            version = pid[1]
            if version > 0:
                raise pickle.UnpicklingError(
                    f"Unsupported TensorizerPickler data version ({version:d})"
                )
            object_type = pid[2]
            if object_type == "storage":
                idx, dtype = pid[3:]
                tensor_view = self.__tensors[idx]
                return self.__tensor_to_storage(tensor_view, dtype)
            else:
                raise pickle.UnpicklingError(
                    f"Unsupported TensorizerPickler object type ({object_type})"
                )
        else:
            # Will probably just throw an error
            return super().persistent_load(pid)

    @staticmethod
    def __wrap_persistent_load(persistent_load_func: callable):

        @functools.wraps(persistent_load_func)
        def _persistent_load(self, pid):
            try:
                return super(self.__class__, self).persistent_load(pid)
            except pickle.UnpicklingError:
                pass
            return persistent_load_func(self, pid)

        return _persistent_load

    def __setattr__(self, key, value):
        if key == "persistent_load":
            value = self.__wrap_persistent_load(value)
        super().__setattr__(key, value)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "persistent_load" in cls.__dict__:
            cls.persistent_load = cls.__wrap_persistent_load(
                cls.persistent_load
            )


def _pickle_attr(name):
    return getattr(pickle, name)


_tensorizer_pickle = types.ModuleType("tensorizer_pickle")
_tensorizer_pickle.__getattr__ = _pickle_attr
_tensorizer_pickle.Pickler = _TensorizerPickler
_tensorizer_pickle.Unpickler = _TensorizerUnpickler


_ORIG_TORCH_SAVE: Final[callable] = torch.save
_ORIG_TORCH_LOAD: Final[callable] = torch.load


def _infer_tensor_ext_name(f: torch.types.FileLike):
    filename: str
    try:
        filename = os.fsdecode(f)
    except TypeError:
        if hasattr(f, "name"):
            filename = os.fsdecode(f.name)
        else:
            raise
    return filename + ".tensors"


@contextlib.contextmanager
def _contextual_torch_filename(f: torch.types.FileLike):
    if _tensorizer_filename.get() is None:
        token = _tensorizer_filename.set(_infer_tensor_ext_name(f))
    elif callable(filename_callback := _tensorizer_filename.get()):
        token = _tensorizer_filename.set(filename_callback(f))
    else:
        token = None
    try:
        yield
    finally:
        if token is not None:
            _tensorizer_filename.reset(token)


_save_wrapper_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_save_wrapper_active", default=False
)
_save_wrapper_active_count: int = 0
_save_wrapper_active_mutex: threading.Lock = threading.Lock()
_save_wrapper_wrapped: Optional[callable] = None

_load_wrapper_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_load_wrapper_active", default=False
)
_load_wrapper_active_count: int = 0
_load_wrapper_active_mutex: threading.Lock = threading.Lock()
_load_wrapper_wrapped: Optional[callable] = None

_suppress_weights_only: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_suppress_weights_only", default=False
)


@functools.wraps(_ORIG_TORCH_SAVE)
def _save_wrapper(
    obj: object,
    f: torch.types.FileLike,
    pickle_module: typing.Any = pickle,
    *args,
    **kwargs,
):
    if not _save_wrapper_active.get():
        return _ORIG_TORCH_SAVE(obj, f, pickle_module, *args, **kwargs)
    if pickle_module is not None and pickle_module is not pickle:
        raise ValueError(
            "Tensorizer-based torch serialization is incompatible with"
            " using a pickle_module other than the default"
        )
    with _contextual_torch_filename(f):
        return _ORIG_TORCH_SAVE(
            obj, f, *args, pickle_module=_tensorizer_pickle, **kwargs
        )


# This signature quietly changed in torch 1.13.0 to default to None,
# but the documentation wasn't updated to reflect that.
_LOAD_WRAPPER_DEFAULT_MODULE: typing.Any = (
    pickle if torch.__version__ < (1, 13, 0) else None
)


@functools.wraps(_ORIG_TORCH_LOAD)
def _load_wrapper(
    f: torch.types.FileLike,
    map_location: torch.serialization.MAP_LOCATION = None,
    pickle_module: typing.Any = _LOAD_WRAPPER_DEFAULT_MODULE,
    *args,
    weights_only: Optional[bool] = None,
    **kwargs,
):
    if not _load_wrapper_active.get():
        return _ORIG_TORCH_LOAD(
            f,
            map_location,
            pickle_module,
            *args,
            weights_only=weights_only,
            **kwargs,
        )
    if pickle_module is not None and pickle_module is not pickle:
        raise ValueError(
            "Tensorizer-based torch serialization is incompatible with"
            " using a pickle_module other than the default"
        )
    with _contextual_torch_filename(f):
        if _suppress_weights_only.get():
            weights_only = False
        return _ORIG_TORCH_LOAD(
            f,
            map_location,
            pickle_module=_tensorizer_pickle,
            *args,
            weights_only=weights_only,
            **kwargs,
        )


@contextlib.contextmanager
def tensorizer_saving(
    file_obj: Optional[_wrapper_file_obj_type] = None, **kwargs
):
    """
    Context manager that modifies calls to ``torch.save`` to use tensorizer
    as a backend for the serialization of tensors and tensor storages.

    Tensors are saved in a sidecar file separate from the ``.pt`` file created
    by ``torch.save``. To load them again, use the `tensorizer_loading`
    context manager paired with ``torch.load``.

    Notes:
        This context manager is thread-safe and async-safe. Other threads or
        coroutines executing concurrently while this context is active will not
        be modified.

    Args:
        file_obj: The file or file-like object in which to save tensor data,
            separate from the one passed to ``torch.save`` for saving metadata.
            This can be any type accepted by a `TensorSerializer`, or a callable
            that dynamically generates the file path or file object based on
            the file path or file-like object ``f`` passed to the ``torch.save``
            call. When using a callable, it should take a single argument of
            the type ``torch.types.FileLike``, and output a type accepted
            by a `TensorSerializer`. The default behaviour is to use a callable
            that appends ``".tensors"`` to any filename passed as ``f``.
        **kwargs: Further keyword arguments to pass to the `TensorSerializer`
            object used to save tensor data.
    """
    global _save_wrapper_active_count, _save_wrapper_wrapped
    active_token = _save_wrapper_active.set(True)
    kwargs_token = _tensorizer_serializer_kwargs.set(kwargs)
    filename_token = _tensorizer_filename.set(file_obj)
    with _save_wrapper_active_mutex:
        _save_wrapper_active_count += 1
        if _save_wrapper_active_count == 1:
            assert _save_wrapper_wrapped is None
            torch.save, _save_wrapper_wrapped = _save_wrapper, torch.save
    try:
        yield
    finally:
        with _save_wrapper_active_mutex:
            _save_wrapper_active_count -= 1
            if _save_wrapper_active_count == 0:
                assert _save_wrapper_wrapped is not None
                torch.save = _save_wrapper_wrapped
                _save_wrapper_wrapped = None
        _tensorizer_filename.reset(filename_token)
        _tensorizer_serializer_kwargs.reset(kwargs_token)
        _save_wrapper_active.reset(active_token)


@contextlib.contextmanager
def tensorizer_loading(
    file_obj: Optional[_wrapper_file_obj_type] = None,
    *,
    suppress_weights_only: bool = False,
    **kwargs,
):
    """
    Context manager that modifies calls to ``torch.load`` to use tensorizer
    as a backend for the deserialization of tensors and tensor storages.
    This is only valid to use when deserializing files that were serialized
    using the corresponding `tensorizer_saving` context manager paired with
    ``torch.save``.

    Tensors are saved in a sidecar file separate from the ``.pt`` file created
    by ``torch.save``. Both must be present at deserialization time.

    Notes:
        This context manager is thread-safe and async-safe. Other threads or
        coroutines executing concurrently while this context is active will not
        be modified.

    Args:
        file_obj: The file or file-like object from which to load tensor data,
            separate from the one passed to ``torch.load`` for loading metadata.
            This can be any type accepted by a `TensorDeserializer`, or a
            callable that dynamically generates the file path or file object
            based on the file path or file-like object `f` passed to the
            ``torch.load`` call. When using a callable, it should take a single
            argument of the type ``torch.types.FileLike``, and output a type
            accepted by a `TensorDeserializer`. The default behaviour is to use
            a callable that appends ``".tensors"`` to any filename passed as
            ``f``.
        **kwargs: Further keyword arguments to pass to the `TensorDeserializer`
            object used to load tensor data.
    """
    global _load_wrapper_active_count, _load_wrapper_wrapped
    active_token = _load_wrapper_active.set(True)
    weights_token = _suppress_weights_only.set(suppress_weights_only)
    kwargs_token = _tensorizer_deserializer_kwargs.set(kwargs)
    filename_token = _tensorizer_filename.set(file_obj)
    with _load_wrapper_active_mutex:
        _load_wrapper_active_count += 1
        if _load_wrapper_active_count == 1:
            assert _load_wrapper_wrapped is None
            torch.load, _load_wrapper_wrapped = _load_wrapper, torch.load
    try:
        yield
    finally:
        with _load_wrapper_active_mutex:
            _load_wrapper_active_count -= 1
            if _load_wrapper_active_count == 0:
                assert _load_wrapper_wrapped is not None
                torch.load = _load_wrapper_wrapped
                _load_wrapper_wrapped = None
        _tensorizer_filename.reset(filename_token)
        _tensorizer_deserializer_kwargs.reset(kwargs_token)
        _suppress_weights_only.reset(weights_token)
        _load_wrapper_active.reset(active_token)
