import tensorflow as tf
import syft


def test_serde_constant():
    z = tf.constant([1.0, 2.0])

    ser = syft.serde.serialize(z)
    x = syft.serde.deserialize(ser)

    assert all(tf.math.equal(x, z))
    assert x.id == z.id
    assert x.dtype == z.dtype


def test_serde_tensorshape():
    z = tf.TensorShape([1, 2])

    ser = syft.serde.serialize(z)
    x = syft.serde.deserialize(ser)

    assert all(tf.math.equal(x, z))


def test_serde_model():

    inp = tf.keras.layers.Input(shape=(3,))
    out = tf.keras.layers.Dense(2)(inp)
    m = tf.keras.Model(inp, out)
    x = tf.constant([[1, 2, 3.0]])
    y = m(x)

    ser = syft.serde.serialize(m)
    m_new = syft.serde.deserialize(ser)

    assert tf.math.equal(y, m_new(x)).numpy().all()
