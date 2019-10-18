import tensorflow as tf
import numpy as np


def test_send_get_variable(remote):
    x_to_give = tf.Variable(2.0)
    x_ptr = x_to_give.send(remote)
    x_gotten = x_ptr.get()
    assert np.array_equal(x_to_give.numpy(), x_gotten.numpy())


def test_add_variables(remote):
  x = tf.Variable([3.0, 3.0]).send(remote)
  y = tf.Variable([2.0, 2.0]).send(remote)
  z_ptr = x + y
  z = z_ptr.get()
  assert np.array_equal(z.numpy(), [5., 5.])

def test_variable_repr(remote):
  x = tf.Variable([1, 2, 3]).send(remote)
  repr = str(x)
  assert repr.startswith('(Wrapper)>[PointerTensor | me')
