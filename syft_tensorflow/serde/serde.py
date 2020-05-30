from collections import OrderedDict
import io
import os
from tempfile import TemporaryDirectory
import zipfile

import syft
from syft.generic.abstract.object import initialize_object
from syft.generic.abstract.tensor import initialize_tensor
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


def _simplify_tf_tensor(tensor: tf.Tensor) -> bin:
    """
    This function converts a TF tensor into a serialized TF tensor using
    tf.io. We do this because it's native to TF, and they've optimized it.

    Args:
      tensor (tf.Tensor): an input tensor to be serialized

    Returns:
      tuple: serialized tuple of TensorFlow tensor. The first value is the
      id of the tensor and the second is the binary for the TensorFlow
      object. The third is the tensor dtype and the fourth is the chain
      of abstractions.
    """

    tensor_ser = tf.io.serialize_tensor(tensor)

    dtype_ser = syft.serde._simplify(tensor.dtype)

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(tensor.child)

    return tensor.id, tensor_ser.numpy(), dtype_ser, chain


def _detail_tf_tensor(worker, tensor_tuple) -> tf.Tensor:
    """
    This function converts a serialized tf tensor into a local TF tensor
    using tf.io.

    Args:
        tensor_tuple (bin): serialized obj of TF tensor. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            TensorFlow object, the third value is the tensor_dtype_enum, and
            the fourth value is the chain of tensor abstractions

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    tensor_id, tensor_bin, tensor_dtype_enum, chain = tensor_tuple

    tensor_dtype = syft.serde._detail(worker, tensor_dtype_enum)
    tensor = tf.io.parse_tensor(tensor_bin, tensor_dtype)

    initialize_tensor(
        hook=syft.tensorflow.hook,
        obj=tensor,
        owner=worker,
        id=tensor_id,
        init_args=[],
        init_kwargs={},
    )

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


def _simplify_tf_variable(tensor: tf.Variable) -> bin:
    """
    This function converts a TF variable into a serialized TF variable using
    tf.io. We do this because it's native to TF, and they've optimized it.

    Args:
      tensor (tf.Variable): an input variable to be serialized

    Returns:
      tuple: serialized tuple of TensorFlow tensor. The first value is the
      id of the tensor and the second is the binary for the TensorFlow
      object. The third is the tensor dtype and the fourth is the chain
      of abstractions.
    """

    tensor_ser = tf.io.serialize_tensor(tensor)

    dtype_ser = syft.serde._simplify(tensor.dtype)

    chain = None
    if hasattr(tensor, "child"):
        chain = syft.serde._simplify(tensor.child)

    return tensor.id, tensor_ser.numpy(), dtype_ser, chain


def _detail_tf_variable(worker, tensor_tuple) -> tf.Tensor:
    """
    This function converts a serialized TF variable into a local TF tensor
    using tf.io.

    Args:
        tensor_tuple (bin): serialized obj of TF variable. It's a tuple where
            the first value is the ID, the second vlaue is the binary for the
            TensorFlow object, the third value is the tensor_dtype_enum, and
            the fourth value is the chain of tensor abstractions

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    tensor_id, tensor_bin, tensor_dtype_enum, chain = tensor_tuple

    tensor_dtype = syft.serde._detail(worker, tensor_dtype_enum)
    tensor = tf.io.parse_tensor(tensor_bin, tensor_dtype)
    tensor = tf.Variable(tensor)

    initialize_tensor(
        hook=syft.tensorflow.hook,
        obj=tensor,
        owner=worker,
        id=tensor_id,
        init_args=[],
        init_kwargs={},
    )

    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensor.child = chain
        tensor.is_wrapper = True

    return tensor


def _simplify_tf_keras_layers(layer: tf.keras.layers.Layer) -> bin:
    """
    This function converts a keras layer into a serialized keras layer using
    keras serialize.

    Args:
      layer (tf.keras.layers.Layer): an input tensor to be serialized

    Returns:
      tuple: serialized tuple of Layer class. The first value is the
      id of the layer and the second is the layer configuration
      object. The third is the layer weights and the fourth is the
      batch input shape.
    """

    layer_ser = tf.keras.layers.serialize(layer)

    weights = layer.get_weights()
    weights_ser = syft.serde.serde._simplify(weights)

    layer_dict_ser = syft.serde.serde._simplify(layer_ser)

    batch_input_shape_ser = syft.serde.serde._simplify(
        layer_ser["config"]["batch_input_shape"]
    )

    return layer.id, layer_dict_ser, weights_ser, batch_input_shape_ser


def _detail_tf_keras_layers(worker, layer_tuple) -> tf.Tensor:
    """
    This function converts a serialized keras layer into a local keras layer

    Args:
        layer_tuple (bin): serialized obj of TF layer. It's a tuple where
            the first value is the ID, the second value is the binary for the
            layer object, the third value is the layer weights, and
            the fourth value is the batch input shape.

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    layer_id, layer_bin, weights_bin, batch_input_shape_bin = layer_tuple

    layer_dict = syft.serde.serde._detail(worker, layer_bin)

    layer = tf.keras.layers.deserialize(layer_dict)

    weights = syft.serde.serde._detail(worker, weights_bin)

    batch_input_shape = syft.serde.serde._detail(worker, batch_input_shape_bin)

    layer.build(batch_input_shape)

    layer.set_weights(weights)

    initialize_object(
        hook=syft.tensorflow.hook,
        obj=layer,
        owner=worker,
        reinitialize=False,
        id=layer_id,
        init_args=[],
        init_kwargs={},
    )

    return layer


def _simplify_keras_model(model: tf.keras.Model):
    """
    This function converts a model into a serialized saved_model.

    Args:
      model (tf.keras.Model: Keras model

    Returns:
      tuple: serialized tuple of Keras model. The first value is
      the binary of the model. The second is the model id.
    """
    bio = io.BytesIO()

    with TemporaryDirectory() as model_location:
        tf.keras.models.save_model(
            model,
            model_location,
            overwrite=True,
            include_optimizer=True,
            save_format='tf',
        )

        with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as model_file:
            # TODO: This process is way inefficient. Find a faster way to serde
            #  SavedModels. See commit b55195a2d4b6852642f987a5a4105167120d123b
            #  for an h5py implementation (which is less general, but clearly
            #  more efficient)
            for root, dir, files in os.walk(model_location):
                relroot = os.path.relpath(root, start=model_location)
                for file in files:
                    filename = os.path.join(root, file)
                    arcname = os.path.join(relroot, file)
                    model_file.write(filename, arcname=arcname)

    model_ser = bio.getvalue()

    return model_ser, model.id


def _detail_keras_model(worker, model_tuple):
    """
    This function converts a serialized model into a local
    model.

    Args:
        modeltuple (bin): serialized obj of Keras model.
        It's a tuple where the first value is the binary of the model.
        The second is the model id.

    Returns:
        tf.keras.models.Model: a deserialized Keras model
    """
    model_ser, model_id = model_tuple
    bio = io.BytesIO(model_ser)
    with TemporaryDirectory() as model_location:
        with zipfile.ZipFile(bio, 'r', zipfile.ZIP_DEFLATED) as model_file:
            # WARNING: zipped archives can potentially deposit extra files onto
            #  the system, although Python's zipfile offers some protection
            #  more info: https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.extractall
            # TODO: further investigate security, find better option if needed
            model_file.extractall(model_location)
        model = tf.keras.models.load_model(model_location)

    initialize_object(
        hook=syft.tensorflow.hook,
        obj=model,
        owner=worker,
        reinitialize=False,
        id=model_id,
        init_args=[],
        init_kwargs={},
    )

    return model


def _simplify_keras_history_callback(
        history_cb: tf.keras.callbacks.History
    ) -> bin:
    """
    This function converts history callback into serialized dictionaries.

    Args:
      history_cb (tf.keras.callbacks.History): history callback

    Returns:
      tuple: serialized tuple of history callback. The first value is
      the binary of the params dictionary. The second is the binary 
      of the history dictionary.
    """

    params = history_cb.params
    history = history_cb.history

    params_ser = syft.serde.serde._simplify(params)
    history_ser = syft.serde.serde._simplify(history)

    return params_ser, history_ser


def _detail_keras_history_callback(
        worker, 
        history_cb_tuple
    ) -> tf.keras.callbacks.History:
    """
    This function converts a serialized history callback into an 
    history callback.

    Args:
        history_cb_tuple (bin): serialized obj of history callback.
        It's a tuple where the first value is the binary of the params 
        dictionary. The second is the binary of the history dictionary.

    Returns:
        tf.keras.callbacks.History: a deserialized history callback
    """

    params_bin, history_bin = history_cb_tuple

    params = syft.serde.serde._detail(worker, params_bin)
    history = syft.serde.serde._detail(worker, history_bin)

    history_cb = tf.keras.callbacks.History()
    history_cb.set_params(params)
    history_cb.history = history

    return history_cb


def _simplify_tf_dtype(dtype: tf.dtypes.DType) -> bin:

    return dtype.as_datatype_enum, None


def _detail_tf_dtype(worker, dtype_tuple):
    dtype_enum, _ = dtype_tuple

    dtype = tf.dtypes.DType(dtype_enum)

    return dtype


def _simplify_tf_tensorshape(tensorshape: tf.TensorShape) -> bin:
    """
    This function converts a TF tensor shape into a serialized list.

    Args:
      tensor (tf.TensorShape): an input tensor shape to be serialized

    Returns:
      tuple: serialized tuple of TF tensor shape. The first value is
      the binary for the TensorShape object. The second is the
      chain of abstractions.
    """

    tensorshape_list_ser = syft.serde.serde._simplify(tensorshape.as_list())

    # TODO[Yann] currently this condition is never true,
    # tf.TensorShape needs to be hooked
    chain = None
    if hasattr(tensorshape, "child"):
        chain = syft.serde._simplify(tensorshape.child)

    return tensorshape_list_ser, chain


def _detail_tf_tensorshape(worker, tensor_tuple) -> tf.TensorShape:
    """
    This function converts a serialized TF tensor shape into a local list.

    Args:
        tensor_tuple (bin): serialized obj of TF tensor shape as list.
            It's a tuple where the first value is the binary for the
            tensorflow shape (as list), and the third value is the
            chain of tensor abstractions.

    Returns:
        tf.Tensor: a deserialized TF tensor
    """

    tensorshape_list_bin, chain = tensor_tuple

    tensorshape_list = syft.serde.serde._detail(worker, tensorshape_list_bin)

    tensorshape = tf.TensorShape(tensorshape_list)

    # TODO[Yann] currently this condition is never true,
    # tf.TensorShape needs to be hooked
    if chain is not None:
        chain = syft.serde._detail(worker, chain)
        tensorshape.child = chain
        tensorshape.is_wrapper = True

    return tensorshape


MAP_TF_SIMPLIFIERS_AND_DETAILERS = OrderedDict(
    {
        EagerTensor: (_simplify_tf_tensor, _detail_tf_tensor),
        tf.Tensor: (_simplify_tf_tensor, _detail_tf_tensor),
        tf.TensorShape: (_simplify_tf_tensorshape, _detail_tf_tensorshape),
        tf.Variable: (_simplify_tf_variable, _detail_tf_variable),
        tf.keras.layers.Layer: (_simplify_tf_keras_layers, _detail_tf_keras_layers),
        tf.keras.models.Model: (_simplify_keras_model, _detail_keras_model),
        ResourceVariable: (_simplify_tf_variable, _detail_tf_variable),
        tf.keras.callbacks.History: (_simplify_keras_history_callback,
                                     _detail_keras_history_callback),
        tf.dtypes.DType: (_simplify_tf_dtype, _detail_tf_dtype),
    }
)
