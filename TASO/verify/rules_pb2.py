# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rules.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='rules.proto',
  package='GraphSubst',
  syntax='proto2',
  serialized_pb=_b('\n\x0brules.proto\x12\nGraphSubst\"\'\n\tParameter\x12\x0b\n\x03key\x18\x01 \x02(\x05\x12\r\n\x05value\x18\x02 \x02(\x05\"$\n\x06Tensor\x12\x0c\n\x04opId\x18\x01 \x02(\x05\x12\x0c\n\x04tsId\x18\x02 \x02(\x05\"`\n\x08Operator\x12\x0c\n\x04type\x18\x01 \x02(\x05\x12!\n\x05input\x18\x02 \x03(\x0b\x32\x12.GraphSubst.Tensor\x12#\n\x04para\x18\x03 \x03(\x0b\x32\x15.GraphSubst.Parameter\"O\n\tMapOutput\x12\x0f\n\x07srcOpId\x18\x01 \x02(\x05\x12\x0f\n\x07\x64stOpId\x18\x02 \x02(\x05\x12\x0f\n\x07srcTsId\x18\x03 \x02(\x05\x12\x0f\n\x07\x64stTsId\x18\x04 \x02(\x05\"}\n\x04Rule\x12#\n\x05srcOp\x18\x01 \x03(\x0b\x32\x14.GraphSubst.Operator\x12#\n\x05\x64stOp\x18\x02 \x03(\x0b\x32\x14.GraphSubst.Operator\x12+\n\x0cmappedOutput\x18\x03 \x03(\x0b\x32\x15.GraphSubst.MapOutput\"0\n\x0eRuleCollection\x12\x1e\n\x04rule\x18\x01 \x03(\x0b\x32\x10.GraphSubst.Rule')
)




_PARAMETER = _descriptor.Descriptor(
  name='Parameter',
  full_name='GraphSubst.Parameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='GraphSubst.Parameter.key', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='GraphSubst.Parameter.value', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=66,
)


_TENSOR = _descriptor.Descriptor(
  name='Tensor',
  full_name='GraphSubst.Tensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='opId', full_name='GraphSubst.Tensor.opId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tsId', full_name='GraphSubst.Tensor.tsId', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=104,
)


_OPERATOR = _descriptor.Descriptor(
  name='Operator',
  full_name='GraphSubst.Operator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='GraphSubst.Operator.type', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input', full_name='GraphSubst.Operator.input', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='para', full_name='GraphSubst.Operator.para', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=106,
  serialized_end=202,
)


_MAPOUTPUT = _descriptor.Descriptor(
  name='MapOutput',
  full_name='GraphSubst.MapOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='srcOpId', full_name='GraphSubst.MapOutput.srcOpId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dstOpId', full_name='GraphSubst.MapOutput.dstOpId', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='srcTsId', full_name='GraphSubst.MapOutput.srcTsId', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dstTsId', full_name='GraphSubst.MapOutput.dstTsId', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=204,
  serialized_end=283,
)


_RULE = _descriptor.Descriptor(
  name='Rule',
  full_name='GraphSubst.Rule',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='srcOp', full_name='GraphSubst.Rule.srcOp', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dstOp', full_name='GraphSubst.Rule.dstOp', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mappedOutput', full_name='GraphSubst.Rule.mappedOutput', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=285,
  serialized_end=410,
)


_RULECOLLECTION = _descriptor.Descriptor(
  name='RuleCollection',
  full_name='GraphSubst.RuleCollection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rule', full_name='GraphSubst.RuleCollection.rule', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=412,
  serialized_end=460,
)

_OPERATOR.fields_by_name['input'].message_type = _TENSOR
_OPERATOR.fields_by_name['para'].message_type = _PARAMETER
_RULE.fields_by_name['srcOp'].message_type = _OPERATOR
_RULE.fields_by_name['dstOp'].message_type = _OPERATOR
_RULE.fields_by_name['mappedOutput'].message_type = _MAPOUTPUT
_RULECOLLECTION.fields_by_name['rule'].message_type = _RULE
DESCRIPTOR.message_types_by_name['Parameter'] = _PARAMETER
DESCRIPTOR.message_types_by_name['Tensor'] = _TENSOR
DESCRIPTOR.message_types_by_name['Operator'] = _OPERATOR
DESCRIPTOR.message_types_by_name['MapOutput'] = _MAPOUTPUT
DESCRIPTOR.message_types_by_name['Rule'] = _RULE
DESCRIPTOR.message_types_by_name['RuleCollection'] = _RULECOLLECTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Parameter = _reflection.GeneratedProtocolMessageType('Parameter', (_message.Message,), dict(
  DESCRIPTOR = _PARAMETER,
  __module__ = 'rules_pb2'
  # @@protoc_insertion_point(class_scope:GraphSubst.Parameter)
  ))
_sym_db.RegisterMessage(Parameter)

Tensor = _reflection.GeneratedProtocolMessageType('Tensor', (_message.Message,), dict(
  DESCRIPTOR = _TENSOR,
  __module__ = 'rules_pb2'
  # @@protoc_insertion_point(class_scope:GraphSubst.Tensor)
  ))
_sym_db.RegisterMessage(Tensor)

Operator = _reflection.GeneratedProtocolMessageType('Operator', (_message.Message,), dict(
  DESCRIPTOR = _OPERATOR,
  __module__ = 'rules_pb2'
  # @@protoc_insertion_point(class_scope:GraphSubst.Operator)
  ))
_sym_db.RegisterMessage(Operator)

MapOutput = _reflection.GeneratedProtocolMessageType('MapOutput', (_message.Message,), dict(
  DESCRIPTOR = _MAPOUTPUT,
  __module__ = 'rules_pb2'
  # @@protoc_insertion_point(class_scope:GraphSubst.MapOutput)
  ))
_sym_db.RegisterMessage(MapOutput)

Rule = _reflection.GeneratedProtocolMessageType('Rule', (_message.Message,), dict(
  DESCRIPTOR = _RULE,
  __module__ = 'rules_pb2'
  # @@protoc_insertion_point(class_scope:GraphSubst.Rule)
  ))
_sym_db.RegisterMessage(Rule)

RuleCollection = _reflection.GeneratedProtocolMessageType('RuleCollection', (_message.Message,), dict(
  DESCRIPTOR = _RULECOLLECTION,
  __module__ = 'rules_pb2'
  # @@protoc_insertion_point(class_scope:GraphSubst.RuleCollection)
  ))
_sym_db.RegisterMessage(RuleCollection)


# @@protoc_insertion_point(module_scope)
