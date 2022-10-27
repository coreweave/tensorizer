import torch
from torch import Tensor
import numpy as np
import tensors.tensors_pb2 as tensors_pb
from typing import OrderedDict

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

def no_init_or_tensor(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy
    original_empty = torch.empty

    torch.empty = lambda *args, **kwargs: original_empty(*args, **{**kwargs, "device": "meta"})

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]
    torch.empty = original_empty

    return result

def serialize(t: Tensor) -> tensors_pb.Tensor:
    assert isinstance(t, Tensor)
    return tensors_pb.Tensor(
        dtype=DtypePbs[t.dtype],
        shape=t.shape,
        data=t.detach().numpy().tobytes(),
    )

def deserialize(t: tensors_pb.Tensor) -> Tensor:
    mv = bytearray(t.data)
    return torch.as_tensor(np.ndarray.__new__(np.memmap,
                           t.shape,
                           dtype=PbNpyDtypes[t.dtype],
                           buffer=mv,
                           offset=0))

def serialize_model(model: torch.nn.Module, filename: str) -> None:
    f = open(filename, "wb")
    modules = list()
    for module_name, module in model.named_modules():
        print(module_name)
        attributes = list()
        for name, param in module.named_parameters(recurse=False):
            v = param.cpu().detach()
            param_attr = tensors_pb.Attribute(
                name=name,
                tensor=tensors_pb.Tensor(
                    dtype=DtypePbs[v.dtype],
                    shape=v.shape,
                    data=v.numpy().tobytes(),
                    attr_type = tensors_pb.AT_PARAMETER
                )
            )
            attributes.append(param_attr)
        for name, buffer in module.named_buffers(recurse=False):
            v = buffer.cpu().detach()
            buffer_attr = tensors_pb.Attribute(
                name=name,
                tensor=tensors_pb.Tensor(
                        dtype=DtypePbs[v.dtype],
                        shape=v.shape,
                        data=v.numpy().tobytes(),
                        attr_type = tensors_pb.AT_BUFFER
                )
            )
            attributes.append(buffer_attr)
        # a little confusing lol
        module_attr = tensors_pb.Attribute(
            name=module_name,
            module=tensors_pb.Module(
                attributes=attributes
            )
        )
        modules.append(module_attr)
    model_proto = tensors_pb.Module( # models are just modules as attributes
        name="",
        attributes=modules,
    )
    f.write(model_proto.SerializeToString())
    f.close()

def deserialize_model(model: torch.nn.Module, filename: str) -> None:
    f = open(filename, "rb")
    model_proto = tensors_pb.Module()
    model_proto.ParseFromString(f.read())
    f.close()

    modules: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for name, module in model.named_modules():
        modules[name] = module
    
    for module_attr in model_proto.attributes:
        module = modules[module_attr.name]
        for attr in module_attr.module.attributes:
            if attr.tensor.attr_type == tensors_pb.AT_PARAMETER:
                module._parameters[attr.name] = deserialize(attr.tensor)
            elif attr.tensor.attr_type == tensors_pb.AT_BUFFER:
                module._buffers[attr.name] = deserialize(attr.tensor)
            else:
                raise ValueError("Unknown attribute type")
