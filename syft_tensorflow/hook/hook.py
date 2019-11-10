import inspect
import logging
import types

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

import syft
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.generic.frameworks.hook.hook import FrameworkHook
from syft.generic.object import initialize_object

from syft_tensorflow.attributes import TensorFlowAttributes
from syft_tensorflow.syft_types import TensorFlowTensor
from syft_tensorflow.syft_types import TensorFlowVariable
from syft_tensorflow.syft_types import KerasLayer
from syft_tensorflow.syft_types import KerasModel


class TensorFlowHook(FrameworkHook):
    def __init__(
        self,
        tensorflow,
        local_worker: BaseWorker = None,
        is_client: bool = True
    ):

        self.tensorflow = tensorflow
        self.framework = self.tensorflow

        syft.tensorflow = TensorFlowAttributes(tf, self)

        syft.framework = syft.tensorflow
        syft.tensorflow.hook = self
        syft.hook = self

        self.local_worker = local_worker

        if hasattr(tensorflow, "tf_hooked"):
            logging.warning("TF was already hooked, skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            tensorflow.tf_hooked = True

        if self.local_worker is None:
            # Every TensorFlowHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the TensorFlow specific code in TensorFlowHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
            )
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {
            tf.math: ["add"],
            Tensor: self._which_methods_should_we_auto_overload(
                Tensor
            ),

            tf.Variable: self._which_methods_should_we_auto_overload(
                tf.Variable
            ),

            tf.keras.layers.Layer: self._which_methods_should_we_auto_overload(
                tf.keras.layers.Layer
            ),

            tf.keras.models.Model: self._which_methods_should_we_auto_overload(
                tf.keras.models.Model
            ),

          ResourceVariable: self._which_methods_should_we_auto_overload(
              ResourceVariable
          ),
        }

        self.args_hook_for_overloaded_attr = {}

        self._hook_native_tensor(Tensor, TensorFlowTensor)
        self._hook_variable(TensorFlowVariable)

        self._hook_keras_layers(tf.keras.layers.Layer, KerasLayer)
        self._hook_keras_model(tf.keras.models.Model, KerasModel)

        self._hook_pointer_tensor_methods(Tensor)
        self._hook_pointer_tensor_methods(tf.Variable)
        self._hook_pointer_tensor_methods(ResourceVariable)

        self._hook_pointer_tensor_methods(tf.math)
        self._hook_multi_pointer_tensor_methods(tf.math)

        self._hook_object_pointer_methods(tf.keras.layers.Layer)
        self._hook_object_pointer_methods(tf.keras.models.Model)

        self._hook_tensorflow_module()

        syft.local_worker = self.local_worker
        syft.hook = self

        # This must happen last!
        # See this functions documentation for more info.
        self._add_methods_to_eager_tensor()

    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Adds PySyft Tensor Functionality to the given native tensor type.
         Overloads the given native TensorFlow tensor to add PySyft Tensor
        Functionality. Overloading involves modifying the tensor type with
        PySyft's added functionality. You may read about what kind of
        modifications are made in the methods that this method calls.
         Args:
            tensor_type: The type of tensor being hooked (in this refactor
                this is only ever tf.Tensor, but in previous versions of
                PySyft this iterated over all tensor types.
            syft_type: The abstract type whose methods should all be added to
                the tensor_type class. In practice this is always TensorFlowTensor.
                Read more about it there.
        """
        # Reinitialize init method of TensorFlow tensor with Syft init
        self._add_registration_to___init__(tensor_type)

        # Overload TensorFlow tensor properties with Syft properties
        self._hook_tensor_properties(tensor_type)

        # Overload auto overloaded with TensorFlow methods
        exclude = [
            "__class__",
            "__delattr__",
            "__dir__",
            "__doc__",
            "__dict__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
            "__eq__",
            "__gt__",
            "__ge__",
            "__lt__",
            "__le__",
        ]
        self._transfer_methods_to_framework_class(tensor_type, syft_type, exclude)

        self._hook_native_methods(tensor_type)

    def _hook_keras_layers(self, layer_cls: type, from_cls: type):

        # Reinitialize init method of the Keras object with Syft init
        self._add_registration_to___init__(layer_cls)

        # Overload Keras object properties with Syft properties
        self._hook_keras_properties(layer_cls)

        # Overload auto overloaded with Keras methods
        exclude = [
            "__class__",
            "__dir__",
            "__doc__",
            "__dict__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
        ]
        self._transfer_methods_to_framework_class(layer_cls, from_cls, exclude)

        self._hook_keras_methods(layer_cls)

    def _hook_keras_model(self, model_cls: type, from_cls: type):

        # Overload the Keras object properties with Syft properties
        self._hook_keras_properties(model_cls)

        # Overload auto overloaded with Keras methods
        exclude = [
            "__class__",
            "__dir__",
            "__doc__",
            "__dict__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
            "__str__",
            "__repr__",
        ]
        self._transfer_methods_to_framework_class(model_cls, from_cls, exclude)

        self._hook_keras_methods(model_cls)

    def _hook_variable(self, syft_type: type):
        """Adds PySyft Tensor functionality to tf.Variable.

        In practice, the user is generally working with subclasses of
        tf.Variable, e.g. ResourceVariable, so we hook methods for those and
        only override the tf.Variable constructor to provide syft registration.
        You may read about what kind of modifications are made in the methods
        that this method calls.

        Args:
            syft_type: The abstract type whose methods should all be added to
                the ResourceVariable class.
        """
        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tf.Variable)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tf.Variable)

        # Overload auto overloaded with Torch methods
        exclude = [
            "__class__",
            "__delattr__",
            "__dict__",
            "__dir__",
            "__doc__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__module__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
        ]
        self._transfer_methods_to_framework_class(ResourceVariable, syft_type, exclude)
        self._hook_properties(ResourceVariable)
        self._hook_native_methods(ResourceVariable)

    def _hook_tensorflow_module(self):
        tensorflow_modules = syft.tensorflow.tensorflow_modules

        for module_name, tensorflow_module in tensorflow_modules.items():
            for func in dir(tensorflow_module):

                # Some functions we want to ignore (not override). Such functions have been hard
                # coded into the tensorflow_attribute exclude (see TensorFlowAttribute class)
                if func in syft.tensorflow.exclude:
                    continue

                # ignore dunder functions
                if "__" in func:
                    continue

                # ignore capitalized func values which are Classes not functinos
                if func[0].isupper():
                    continue

                # ignore hidden functins
                if func[0] == "_":
                    continue

                # If we haven't already overloaded this function
                if "native_" in func or f"native_{func}" in dir(tensorflow_module):
                    continue

                self._perform_function_overloading(module_name, tensorflow_module, func)

    def _add_registration_to___init__(
        hook_self, tensor_type: type, is_tensor: bool = False
    ):
        """Adds several attributes to the tensor.
         Overloads tensor_type.__init__ to add several attributes to the tensor
        as well as registering the tensor automatically.
         Args:
            tensor_type: The type of tensor being hooked (in this refactor this
                is only ever tf.Tensor, but in previous versions of PySyft
                this iterated over all tensor types.
            is_tensor: An optional boolean parameter (default False) to
                specify whether to skip running the native initialization
                logic.
        """

        def new___init__(self, *args, owner=None, id=None, register=True, **kwargs):
            return initialize_object(
                hook=hook_self,
                obj=self,
                reinitialize=not is_tensor,
                owner=owner,
                id=id,
                init_args=args,
                init_kwargs=kwargs,
            )

        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        tensor_type.__init__ = new___init__

    def _hook_keras_properties(hook_self, keras_type: type):
        super()._hook_properties(keras_type)

    def _hook_tensor_properties(hook_self, tensor_type: type):
        super()._hook_properties(tensor_type)
        tensor_type.native_shape = tensor_type.shape

    def _add_methods_to_eager_tensor(self):
      """
      Add required TensorFlowTensor methods to EagerTensor.

      When a user creates a tensor, e.g. with `tf.constant([1,2,3])`, the
      type of the returned object is an `EagerTensor`.  EagerTensor is defined
      in tensorflow and is a super class of tf.Tensor.  However, EagerTensor is
      not importable so we cannot add the properties to it that we want.
      So we do that here, which requires us to instantiate an instance of an
      EagerTensor to then get reference to the type.

      We do it this way for 2 reasons:

      1. Avoid monkeypatching functions per instance (e.g. having to overwrite the
         function every time we make a tensor)
      2. Most dunder methods are actually looked up on the class itself, rather
         than the instance, so monkeypatching the instance doesn't even work for
         things like __repr__.
      """

      dummy = tf.constant(0)
      eager_type = type(dummy)

      eager_type.native___str__ = eager_type.__str__
      eager_type.native___repr__ = eager_type.__repr__

      eager_type.__repr__ = TensorFlowTensor.__repr__
      eager_type.__str__ = TensorFlowTensor.__str__

      for method in self.to_auto_overload[tf.math]:
          setattr(eager_type, method, getattr(tf, method))

    def _hook_keras_methods(self, keras_type: type):
        """
        Add hooked version of all methods of to_auto_overload[keras_type]
        to the keras_type; instead of performing the native keras object
        method, the hooked version will be called

        Args:
            keras_type: the keras_type which holds the methods
        """

        for attr in self.to_auto_overload[keras_type]:
            # if we haven't already overloaded this function
            if f"native_{attr}" not in dir(keras_type):
                native_method = getattr(keras_type, attr)
                setattr(keras_type, f"native_{attr}", native_method)
                new_method = self._get_hooked_method(attr)
                setattr(keras_type, attr, new_method)

    @classmethod
    def create_shape(cls, shape_dims):
        return tf.TensorShape(shape_dims)

    @classmethod
    def create_zeros(shape, dtype, **kwargs):
        return tf.zeros(shape, dtype=dtype, **kwargs)
