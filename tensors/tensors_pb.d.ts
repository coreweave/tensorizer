// package: tensors
// file: tensors.proto

import * as jspb from "google-protobuf";

export class Tensor extends jspb.Message {
  getDtype(): DtypeMap[keyof DtypeMap];
  setDtype(value: DtypeMap[keyof DtypeMap]): void;

  clearShapeList(): void;
  getShapeList(): Array<number>;
  setShapeList(value: Array<number>): void;
  addShape(value: number, index?: number): number;

  getData(): Uint8Array | string;
  getData_asU8(): Uint8Array;
  getData_asB64(): string;
  setData(value: Uint8Array | string): void;

  hasAttrType(): boolean;
  clearAttrType(): void;
  getAttrType(): AttributeTypeMap[keyof AttributeTypeMap];
  setAttrType(value: AttributeTypeMap[keyof AttributeTypeMap]): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Tensor.AsObject;
  static toObject(includeInstance: boolean, msg: Tensor): Tensor.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Tensor, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Tensor;
  static deserializeBinaryFromReader(message: Tensor, reader: jspb.BinaryReader): Tensor;
}

export namespace Tensor {
  export type AsObject = {
    dtype: DtypeMap[keyof DtypeMap],
    shapeList: Array<number>,
    data: Uint8Array | string,
    attrType: AttributeTypeMap[keyof AttributeTypeMap],
  }
}

export class Attribute extends jspb.Message {
  getName(): string;
  setName(value: string): void;

  hasModule(): boolean;
  clearModule(): void;
  getModule(): Module | undefined;
  setModule(value?: Module): void;

  hasTensor(): boolean;
  clearTensor(): void;
  getTensor(): Tensor | undefined;
  setTensor(value?: Tensor): void;

  hasString(): boolean;
  clearString(): void;
  getString(): string;
  setString(value: string): void;

  hasInt64(): boolean;
  clearInt64(): void;
  getInt64(): number;
  setInt64(value: number): void;

  hasFloat(): boolean;
  clearFloat(): void;
  getFloat(): number;
  setFloat(value: number): void;

  hasBool(): boolean;
  clearBool(): void;
  getBool(): boolean;
  setBool(value: boolean): void;

  getValueCase(): Attribute.ValueCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Attribute.AsObject;
  static toObject(includeInstance: boolean, msg: Attribute): Attribute.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Attribute, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Attribute;
  static deserializeBinaryFromReader(message: Attribute, reader: jspb.BinaryReader): Attribute;
}

export namespace Attribute {
  export type AsObject = {
    name: string,
    module?: Module.AsObject,
    tensor?: Tensor.AsObject,
    string: string,
    int64: number,
    pb_float: number,
    bool: boolean,
  }

  export enum ValueCase {
    VALUE_NOT_SET = 0,
    MODULE = 3,
    TENSOR = 4,
    STRING = 5,
    INT64 = 6,
    FLOAT = 7,
    BOOL = 8,
  }
}

export class Module extends jspb.Message {
  getName(): string;
  setName(value: string): void;

  clearNamesList(): void;
  getNamesList(): Array<string>;
  setNamesList(value: Array<string>): void;
  addNames(value: string, index?: number): string;

  clearAttributesList(): void;
  getAttributesList(): Array<Attribute>;
  setAttributesList(value: Array<Attribute>): void;
  addAttributes(value?: Attribute, index?: number): Attribute;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Module.AsObject;
  static toObject(includeInstance: boolean, msg: Module): Module.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Module, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Module;
  static deserializeBinaryFromReader(message: Module, reader: jspb.BinaryReader): Module;
}

export namespace Module {
  export type AsObject = {
    name: string,
    namesList: Array<string>,
    attributesList: Array<Attribute.AsObject>,
  }
}

export interface DtypeMap {
  DT_INVALID: 0;
  DT_FLOAT32: 1;
  DT_FLOAT64: 2;
  DT_FLOAT16: 3;
  DT_BFLOAT16: 4;
  DT_COMPLEX32: 5;
  DT_COMPLEX64: 6;
  DT_COMPLEX128: 7;
  DT_UINT8: 8;
  DT_INT8: 9;
  DT_INT16: 10;
  DT_INT32: 11;
  DT_INT64: 12;
  DT_BOOL: 13;
  DT_QUINT8: 14;
  DT_QINT8: 15;
  DT_QINT32: 16;
  DT_QUINT4_2: 17;
}

export const Dtype: DtypeMap;

export interface AttributeTypeMap {
  AT_PARAMETER: 0;
  AT_BUFFER: 1;
}

export const AttributeType: AttributeTypeMap;

