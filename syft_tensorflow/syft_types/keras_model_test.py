import pytest
import tensorflow as tf
import syft
import numpy as np


def test_send_get_keras_model(remote):

    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, input_shape=[2])
    ])

    model_ptr = model_to_give.send(remote)
    model_gotten = model_ptr.get()

    assert np.array_equal(model_to_give.get_weights()[0],
                model_gotten.get_weights()[0])


def test_keras_sequential(remote):

    x_to_give = tf.random.uniform([2, 2])
    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, input_shape=[2])
    ])
    expected = model_to_give(x_to_give)

    x_ptr = x_to_give.send(remote)
    model_ptr = model_to_give.send(remote)
    actual = model_ptr(x_ptr).get()

    assert np.array_equal(actual, expected)


def test_keras_model_compile(remote):

    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, input_shape=[2])
    ])

    model_ptr = model_to_give.send(remote)

    model_ptr.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model_on_worker = remote._objects[model_ptr.id_at_location]

    assert model_on_worker.loss == 'categorical_crossentropy'
    assert isinstance(model_on_worker.optimizer,
            tf.keras.optimizers.Adam)


def test_keras_model_fit(remote):

    x_to_give = tf.random.uniform([2, 2], seed=1)
    y_to_give = tf.ones([2, 1])

    k_init = tf.keras.initializers.RandomNormal(seed=1)

    model_to_give = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(5, 
                                          input_shape=[2],
                                          kernel_initializer=k_init)
    ])

    model_to_give.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    

    model_ptr = model_to_give.send(remote)
    x_ptr = x_to_give.send(remote)
    y_ptr = y_to_give.send(remote)

    history_cb = model_ptr.fit(x_ptr, y_ptr, epochs=1)
    final_loss = history_cb.history['loss'][0]

    np.testing.assert_almost_equal(final_loss, 34.6580, decimal=4)



