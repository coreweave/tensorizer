import torch
from torch import Tensor
import numpy as np
import gooseai.tensors.tensors_pb2 as tensors_pb

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
    tensors_pb.DT_BOOL: np.bool,
    tensors_pb.DT_QUINT8: np.uint8,
    tensors_pb.DT_QINT8: np.int8,
    tensors_pb.DT_QINT32: np.int32,
    tensors_pb.DT_QUINT4_2: np.uint8,
}


def serialize(t: Tensor) -> tensors_pb.Tensor:
    assert isinstance(t, Tensor)
    return tensors_pb.Tensor(
        dtype=DtypePbs[t.dtype],
        shape=t.shape,
        data=t.numpy().tobytes(),
    )


def deserialize(t: tensors_pb.Tensor) -> Tensor:
    mv = bytearray(t.data)
    return torch.as_tensor(np.ndarray.__new__(np.memmap,
                           t.shape,
                           dtype=PbNpyDtypes[t.dtype],
                           buffer=mv,
                           offset=0))