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
import functools
import inspect
import io
import logging
import os
import pickle
import threading
import types
import typing
from contextvars import ContextVar
from typing import Any, Callable, Final, Iterable, List, Optional, Tuple, Union

import torch

from .serialization import TensorDeserializer, TensorSerializer

__all__ = (
    "tensorizer_saving",
    "tensorizer_loading",
)

logger = logging.getLogger(__name__)

_tensorizer_file_obj_type: "typing.TypeAlias" = Union[
    io.BufferedIOBase,
    io.RawIOBase,
    typing.BinaryIO,
    str,
    bytes,
    os.PathLike,
    int,
]

_FileLike: "typing.TypeAlias" = Union[str, os.PathLike[str], typing.IO[bytes]]

_wrapper_file_obj_type: "typing.TypeAlias" = Union[
    _tensorizer_file_obj_type,
    Callable[[_FileLike], _tensorizer_file_obj_type],
]

_save_func_type: "typing.TypeAlias" = Callable[
    [_tensorizer_file_obj_type, Iterable[torch.Tensor], dict],
    Any,
]

_load_func_type: "typing.TypeAlias" = Callable[
    [_tensorizer_file_obj_type, dict], Iterable[torch.Tensor]
]

_storage_type: "typing.TypeAlias" = Union[
    torch.UntypedStorage, torch.TypedStorage
]

_tensorizer_loading_filename: ContextVar[Optional[_wrapper_file_obj_type]] = (
    ContextVar("_tensorizer_loading_filename", default=None)
)
_tensorizer_saving_filename: ContextVar[Optional[_wrapper_file_obj_type]] = (
    ContextVar("_tensorizer_saving_filename", default=None)
)

_tensorizer_deserializer_kwargs: ContextVar[Optional[dict]] = ContextVar(
    "_tensorizer_deserializer_kwargs", default=None
)

_tensorizer_serializer_kwargs: ContextVar[Optional[dict]] = ContextVar(
    "_tensorizer_serializer_kwargs", default=None
)


def _storage_device(storage: _storage_type) -> torch.device:
    if isinstance(storage, torch.TypedStorage):
        return getattr(storage, "_untyped_storage", storage).device
    else:
        return storage.device


def _has_data(storage: _storage_type) -> bool:
    maybe_untyped = getattr(storage, "_untyped_storage", storage)
    return maybe_untyped.device.type != "meta" and maybe_untyped.data_ptr() != 0


class _TensorizerPickler(pickle.Pickler):
    __filename: Optional[_tensorizer_file_obj_type]
    __tensors: List[torch.Tensor]
    __tensor_ids: typing.Dict[Tuple[typing.Hashable, ...], int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__filename = _tensorizer_saving_filename.get()
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
                self.__tensor_ids.clear()
                return
            kwargs = _tensorizer_serializer_kwargs.get()
            if kwargs is None:
                kwargs = {}
            try:
                if (save_func := _save_wrapper_save_func.get()) is None:
                    serializer = TensorSerializer(self.__filename, **kwargs)
                    serializer.write_state_dict(self.__tensors)
                    serializer.close()
                else:
                    save_func(self.__filename, self.__tensors, kwargs)
            finally:
                # Don't call .clear() on self.__tensors in case it was saved
                # somewhere by save_func
                self.__tensors = []
                self.__tensor_ids.clear()

    @staticmethod
    def __storage_to_tensor(storage: _storage_type) -> torch.Tensor:
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
            and torch.is_storage(obj)
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
    __has_tensors: bool
    __tensors: Optional[list]
    __cached_super_load: Optional[callable]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__filename = _tensorizer_loading_filename.get()
        self.__has_tensors = self.__filename is not None
        self.__tensors = None
        self.__cached_super_load = None

    def load(self):
        try:
            return super().load()
        finally:
            if self.__tensors is not None:
                self.__tensors.clear()
                self.__tensors = None

    def __load_tensors(self) -> None:
        # Load and cache tensors from a sidecar file
        if self.__tensors is not None:
            return
        elif not self.__has_tensors:
            raise RuntimeError("Tried to load tensors without a path")
        kwargs = _tensorizer_deserializer_kwargs.get()
        if kwargs is None:
            kwargs = {}
        if (load_func := _load_wrapper_load_func.get()) is None:
            with TensorDeserializer(self.__filename, **kwargs) as deserializer:
                self.__tensors = deserializer.tree()
        else:
            self.__tensors = list(load_func(self.__filename, kwargs))
        assert self.__tensors is not None

    @staticmethod
    def __tensor_to_storage(
        tensor: torch.Tensor, dtype: Optional[torch.dtype]
    ) -> _storage_type:
        # Convert a tensor into an equivalent storage
        # for compatibility with a TensorDeserializer
        if dtype is None:
            # Oddly, PyTorch expects a storage serialized as an UntypedStorage
            # to be loaded as a TypedStorage with the torch.uint8 type.
            dtype = torch.uint8
        return torch.TypedStorage(
            wrap_storage=tensor.untyped_storage(), dtype=dtype, _internal=True
        )

    def __get_storage(self, idx: int, dtype: Optional[torch.dtype]):
        # This will load all tensors the first time a "TensorizerPickler"
        # persistent_id is encountered, indicating that this was a file
        # created by a _TensorizerPickler. Deferring it to this point
        # will avoid trying to engage the load logic on .pt files
        # that were NOT created by a _TensorizerPickler, where there
        # is probably no corresponding .tensors file anyway, where trying
        # to load that would fail.
        if self.__tensors is None:
            self.__load_tensors()
        tensor_view = self.__tensors[idx]
        return self.__tensor_to_storage(tensor_view, dtype)

    @property
    def __super_load(self) -> Callable[[Any], Any]:
        if self.__cached_super_load is not None:
            return self.__cached_super_load
        super_load = super().persistent_load
        super_load_func = getattr(super_load, "__func__", super_load)
        # Evil Python behaviour can make the super method equal this method
        # prior to Python 3.13, so check for that to avoid accidental recursion.
        # _is_load_wrapper is set on dynamically-created wrappers
        # that ultimately recurse back to this function; avoid those too.
        if super_load_func == _TensorizerUnpickler.persistent_load or getattr(
            super_load_func, "_is_load_wrapper", False
        ):
            # To avoid recursing forever, just raise the
            # default error from pickle.Unpickler instead
            self.__cached_super_load = self.__fallback_super_load
        else:
            # Will probably just throw an error,
            # but could redirect to a sibling class
            self.__cached_super_load = super_load
        return self.__cached_super_load

    @staticmethod
    def __fallback_super_load(_pid):
        raise pickle.UnpicklingError("unsupported persistent id encountered")

    def persistent_load(self, pid):
        if (
            self.__has_tensors
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
                return self.__get_storage(idx, dtype)
            else:
                raise pickle.UnpicklingError(
                    f"Unsupported TensorizerPickler object type ({object_type})"
                )
        else:
            return self.__super_load(pid)

    @staticmethod
    def __wrap_persistent_load(persistent_load_func: callable):

        @functools.wraps(persistent_load_func)
        def _persistent_load(self, pid):
            try:
                if self.__class__ is _TensorizerUnpickler:
                    # For instances of this class, call this class's method
                    return self.__class__.persistent_load(self, pid)
                else:
                    # For subclasses, defer to the super method
                    return super(self.__class__, self).persistent_load(pid)
            except pickle.UnpicklingError:
                pass
            # This is being set on an instance, not the class,
            # so this wouldn't expect to be passed self as well,
            # as it is not an unbound method here
            return persistent_load_func(pid)

        return _persistent_load

    def __setattr__(self, key, value):
        if key == "persistent_load":
            # If this method is being overridden dynamically, modify it
            # to defer to the persistent_load method from this class first
            wrapped_func = self.__wrap_persistent_load(value)
            # Mark this as a wrapper for recursion detection later on
            wrapped_func._is_load_wrapper = True
            value = types.MethodType(wrapped_func, self)
            # Necessary witchcraft prior to Python 3.13:
            # pickle.Unpickler may internally cache persistent_load functions,
            # and it would normally update the cached value using a PyGetSetDef
            # descriptor, but having a class in the inheritance hierarchy
            # that defines persistent_load as a non-descriptor prevents
            # attribute updates from reaching that descriptor's set method,
            # so the cached value that the unpickler actually uses isn't
            # properly updated, even though the Python object shows it as being
            # updated. We can force this update to propagate to that descriptor
            # by manipulating it directly.
            if (
                pickle.Unpickler in self.__class__.__mro__
                and inspect.isgetsetdescriptor(pickle.Unpickler.persistent_load)
            ):
                pickle.Unpickler.persistent_load.__set__(self, value)
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


def _infer_tensor_ext_name(f: _FileLike):
    if isinstance(f, io.BytesIO):
        logger.warning(
            "Cannot infer .tensors location from io.BytesIO;"
            " not using tensorizer backend"
            " (set the file_obj parameter to choose a location instead)"
        )
        return None
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
def _contextual_torch_filename(
    f: _FileLike,
    filename_ctx_var: ContextVar[Optional[_wrapper_file_obj_type]],
):
    if filename_ctx_var.get() is None:
        token = filename_ctx_var.set(_infer_tensor_ext_name(f))
    elif callable(filename_callback := filename_ctx_var.get()):
        token = filename_ctx_var.set(filename_callback(f))
    else:
        token = None
    try:
        yield
    finally:
        if token is not None:
            filename_ctx_var.reset(token)


_save_wrapper_active: ContextVar[bool] = ContextVar(
    "_save_wrapper_active", default=False
)
_save_wrapper_active_count: int = 0
_save_wrapper_active_mutex: threading.Lock = threading.Lock()
_save_wrapper_wrapped: Optional[callable] = None
_save_wrapper_save_func: ContextVar[Optional[_save_func_type]] = ContextVar(
    "_save_wrapper_save_func", default=None
)

_load_wrapper_active: ContextVar[bool] = ContextVar(
    "_load_wrapper_active", default=False
)
_load_wrapper_active_count: int = 0
_load_wrapper_active_mutex: threading.Lock = threading.Lock()
_load_wrapper_wrapped: Optional[callable] = None
_load_wrapper_load_func: ContextVar[Optional[_load_func_type]] = ContextVar(
    "_load_wrapper_load_func", default=None
)

_suppress_weights_only: ContextVar[bool] = ContextVar(
    "_suppress_weights_only", default=False
)


@functools.wraps(_ORIG_TORCH_SAVE)
def _save_wrapper(
    obj: object,
    f: _FileLike,
    pickle_module: Any = pickle,
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
    with _contextual_torch_filename(f, _tensorizer_saving_filename):
        return _ORIG_TORCH_SAVE(
            obj, f, *args, pickle_module=_tensorizer_pickle, **kwargs
        )


# This signature quietly changed in torch 1.13.0 to default to None,
# but the documentation wasn't updated to reflect that.
_LOAD_WRAPPER_DEFAULT_MODULE: Any = (
    pickle if torch.__version__ < (1, 13, 0) else None
)


@functools.wraps(_ORIG_TORCH_LOAD)
def _load_wrapper(
    f: _FileLike,
    map_location: torch.serialization.MAP_LOCATION = None,
    pickle_module: Any = _LOAD_WRAPPER_DEFAULT_MODULE,
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
    with _contextual_torch_filename(f, _tensorizer_loading_filename):
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
    file_obj: Optional[_wrapper_file_obj_type] = None,
    *,
    save_func: Optional[_save_func_type] = None,
    **kwargs,
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
            If a provided callable returns ``None``, tensorizer deserialization
            is not used.
        save_func: An optional callable with the signature
            ``save_func(file_obj, tensors: Iterable[Tensor], kwargs: dict)``
            that may be used to override the default saving logic for tensors.
            `file_obj` and `kwargs` correspond to the ones passed to this
            function. This may be used, for instance, to make serialization
            asynchronous by writing a `save_func` that serializes in
            a background thread or process.
        kwargs: Further keyword arguments to pass to the `TensorSerializer`
            object used to save tensor data.
    """
    global _save_wrapper_active_count, _save_wrapper_wrapped
    active_token = _save_wrapper_active.set(True)
    kwargs_token = _tensorizer_serializer_kwargs.set(kwargs)
    filename_token = _tensorizer_saving_filename.set(file_obj)
    save_func_token = _save_wrapper_save_func.set(save_func)
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
        _save_wrapper_save_func.reset(save_func_token)
        _tensorizer_saving_filename.reset(filename_token)
        _tensorizer_serializer_kwargs.reset(kwargs_token)
        _save_wrapper_active.reset(active_token)


@contextlib.contextmanager
def tensorizer_loading(
    file_obj: Optional[_wrapper_file_obj_type] = None,
    *,
    load_func: Optional[_load_func_type] = None,
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
            ``f``. If a provided callable returns ``None``, tensorizer
            serialization is not used.
        load_func: An optional callable with the signature
            ``load_func(file_obj, kwargs: dict) -> Iterable[Tensor]``
            that may be used to override the default loading logic for tensors.
            `file_obj` and `kwargs` correspond to the ones passed to this
            function.
        suppress_weights_only: If set to ``True``, replace ``weights_only=True``
            with ``weights_only=False`` in calls to ``torch.load`` within this
            context. Using ``torch.load`` with tensorizer as a backend is
            incompatible with ``weights_only=True`` because ``torch`` counts it
            using a custom ``pickle_module`` as being a non-weights-only load,
            even though tensorizer only loads weights in practice.
        kwargs: Further keyword arguments to pass to the `TensorDeserializer`
            object used to load tensor data.
    """
    global _load_wrapper_active_count, _load_wrapper_wrapped
    active_token = _load_wrapper_active.set(True)
    weights_token = _suppress_weights_only.set(suppress_weights_only)
    kwargs_token = _tensorizer_deserializer_kwargs.set(kwargs)
    filename_token = _tensorizer_loading_filename.set(file_obj)
    load_func_token = _load_wrapper_load_func.set(load_func)
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
        _load_wrapper_load_func.reset(load_func_token)
        _tensorizer_loading_filename.reset(filename_token)
        _tensorizer_deserializer_kwargs.reset(kwargs_token)
        _suppress_weights_only.reset(weights_token)
        _load_wrapper_active.reset(active_token)
