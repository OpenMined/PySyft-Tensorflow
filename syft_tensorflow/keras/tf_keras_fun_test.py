import pytest
import tensorflow as tf

import syft


def test_keras_activations_fn(remote):

    x_to_give = tf.constant([-2.0, 3.0, 5.0])
    expected = tf.keras.activations.relu(x_to_give)

    x_ptr = x_to_give.send(remote)

    relu_ptr = tf.keras.activations.relu(x_ptr)
    actual = relu_ptr.get()

    assert tf.math.equal(actual, expected).numpy().all()

def test_keras_sigmoid(remote):

  expected = tf.keras.activations.sigmoid(tf.constant([1.0, 1.0]))

  x = tf.constant([1.0, 1.0]).send(remote)
  y = tf.keras.activations.sigmoid(x).get()

  assert tf.math.equal(y, expected).numpy().all()
