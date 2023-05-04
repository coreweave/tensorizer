from collections import OrderedDict
from typing import BinaryIO
from typing import OrderedDict as OrderedDictType
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

import tensorizer.tensors_pb2 as tensors_pb
from tensorizer.tensors_pb2 import Tensor as TensorPb

DtypePbs = {
    torch.float32: tensors_pb.DT_FLOAT32,
    torch.float64: tensors_pb.DT_FLOAT64,
    torch.float16: tensors_pb.DT_FLOAT16,
    torch.bfloat16: tensors_pb.DT_BFLOAT16,
    torch.complex32: tensors_pb.DT_COMPLEX32,
    torch.complex64: tensors_pb.DT_COMPLEX64,
    torch.complex128: tensors_pb.DT_COMPLEX128,
    torch.uint8: tensors_pb.DT_UINT8,
    torch.int8: tensors_pb.DT_INT8,
    torch.int16: tensors_pb.DT_INT16,
    torch.int32: tensors_pb.DT_INT32,
    torch.int64: tensors_pb.DT_INT64,
    torch.bool: tensors_pb.DT_BOOL,
    torch.quint8: tensors_pb.DT_QUINT8,
    torch.qint8: tensors_pb.DT_QINT8,
    torch.qint32: tensors_pb.DT_QINT32,
    torch.quint4x2: tensors_pb.DT_QUINT4_2,
}

PbDtypes = {
    tensors_pb.DT_FLOAT32: torch.float32,
    tensors_pb.DT_FLOAT64: torch.float64,
    tensors_pb.DT_FLOAT16: torch.float16,
    tensors_pb.DT_BFLOAT16: torch.bfloat16,
    tensors_pb.DT_COMPLEX32: torch.complex32,
    tensors_pb.DT_COMPLEX64: torch.complex64,
    tensors_pb.DT_COMPLEX128: torch.complex128,
    tensors_pb.DT_UINT8: torch.uint8,
    tensors_pb.DT_INT8: torch.int8,
    tensors_pb.DT_INT16: torch.int16,
    tensors_pb.DT_INT32: torch.int32,
    tensors_pb.DT_INT64: torch.int64,
    tensors_pb.DT_BOOL: torch.bool,
    tensors_pb.DT_QUINT8: torch.quint8,
    tensors_pb.DT_QINT8: torch.qint8,
    tensors_pb.DT_QINT32: torch.qint32,
    tensors_pb.DT_QUINT4_2: torch.quint4x2,
}

PbNpyDtypes = {
    tensors_pb.DT_FLOAT32: np.float32,
    tensors_pb.DT_FLOAT64: np.float64,
    tensors_pb.DT_FLOAT16: np.float16,
    tensors_pb.DT_BFLOAT16: np.float16,
    tensors_pb.DT_COMPLEX32: np.complex64,
    tensors_pb.DT_COMPLEX64: np.complex64,
    tensors_pb.DT_COMPLEX128: np.complex128,
    tensors_pb.DT_UINT8: np.uint8,
    tensors_pb.DT_INT8: np.int8,
    tensors_pb.DT_INT16: np.int16,
    tensors_pb.DT_INT32: np.int32,
    tensors_pb.DT_INT64: np.int64,
    tensors_pb.DT_BOOL: bool,
    tensors_pb.DT_QUINT8: np.uint8,
    tensors_pb.DT_QINT8: np.int8,
    tensors_pb.DT_QINT32: np.int32,
    tensors_pb.DT_QUINT4_2: np.uint8,
}


def serialize_tensor(
    t: Tensor, attribute: tensors_pb.AttributeType = None
) -> tensors_pb.Tensor:
    assert isinstance(t, Tensor)
    assert attribute is None or attribute in [
        tensors_pb.AT_PARAMETER,
        tensors_pb.AT_BUFFER,
    ]

    extra_opts = {}
    if attribute is not None:
        extra_opts = {"attr_type": attribute}

    return tensors_pb.Tensor(
        dtype=DtypePbs[t.dtype],
        shape=t.shape,
        data=t.cpu().detach().numpy().tobytes(),
        **extra_opts,
    )


def deserialize_tensor(
    t: tensors_pb.Tensor,
) -> Union[Tensor, Tuple[Tensor, "tensors_pb.AttributeType"]]:
    mv = bytearray(t.data)
    tensor = torch.as_tensor(
        np.ndarray.__new__(
            np.memmap, t.shape, dtype=PbNpyDtypes[t.dtype], buffer=mv, offset=0
        )
    )
    if t.HasField("attr_type"):
        return tensor, t.attr_type
    else:
        return tensor


def serialize_model(model: torch.nn.Module, file_stream: BinaryIO) -> None:
    modules = list()
    for module_name, module in model.named_modules():
        print(module_name)
        attributes = list()
        for name, param in module.named_parameters(recurse=False):
            v = param.cpu().detach()
            param_attr = tensors_pb.Attribute(
                name=name, tensor=serialize_tensor(v, tensors_pb.AT_PARAMETER)
            )
            attributes.append(param_attr)
        for name, buffer in module.named_buffers(recurse=False):
            v = buffer.cpu().detach()
            buffer_attr = tensors_pb.Attribute(
                name=name, tensor=serialize_tensor(v, tensors_pb.AT_BUFFER)
            )
            attributes.append(buffer_attr)
        module_attr = tensors_pb.Attribute(
            name=module_name, module=tensors_pb.Module(attributes=attributes)
        )
        modules.append(module_attr)
    model_proto = tensors_pb.Module(  # models are just modules as attributes
        name="",
        attributes=modules,
    )
    file_stream.write(model_proto.SerializeToString())


def deserialize_model(model: torch.nn.Module, file_stream: BinaryIO) -> None:
    model_proto = tensors_pb.Module()
    model_proto.ParseFromString(file_stream.read())

    modules: OrderedDictType[str, torch.nn.Module] = OrderedDict()
    for name, module in model.named_modules():
        modules[name] = module

    for module_attr in model_proto.attributes:
        module = modules[module_attr.name]
        for attr in module_attr.module.attributes:
            if attr.tensor.HasField("attr_type"):
                if attr.tensor.attr_type == tensors_pb.AT_PARAMETER:
                    module._parameters[attr.name] = deserialize_tensor(
                        attr.tensor
                    )[0]
                elif attr.tensor.attr_type == tensors_pb.AT_BUFFER:
                    module._buffers[attr.name] = deserialize_tensor(
                        attr.tensor
                    )[0]
                else:
                    raise ValueError("Unknown attribute type")
