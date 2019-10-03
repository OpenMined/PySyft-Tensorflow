"""Hook args implementation for TensorFLow.
Implements and registers hook_args functionality for syft-tensorflow objects.
See syft/generic/frameworks/hook/hook_args.py for the core implementation.
"""

import tensorflow as tf

from tensorflow.python.framework.ops import EagerTensor
from syft.exceptions import PureFrameworkTensorFoundError
from syft.generic.frameworks.hook.hook_args import (
    register_ambiguous_method,
    register_backward_func,
    register_forward_func,
    register_type_rule,
    one,
)

from syft_tensorflow.tensor import TensorFlowTensor


type_rule = {
    tf.Tensor: one,
    TensorFlowTensor: one,
    EagerTensor: one,
}

def default_forward(i):
  if hasattr(i, "child"):
    return i.child

  return (_ for _ in ()).throw(PureFrameworkTensorFoundError)

forward_func = {
    tf.Tensor: default_forward,
    EagerTensor: default_forward,
}
backward_func = {
    tf.Tensor: lambda i: i.wrap(),
    TensorFlowTensor: lambda i: i.wrap(),
    EagerTensor: lambda i: i.wrap(),
}
ambiguous_methods = {"__getitem__", "__setitem__"}

register_ambiguous_method(*ambiguous_methods)
register_type_rule(type_rule)
register_forward_func(forward_func)
register_backward_func(backward_func)
