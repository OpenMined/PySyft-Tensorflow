import weakref

import tensorflow as tf

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.object import AbstractObject
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker

from syft_tensorflow.syft_types import KerasObject

from syft.exceptions import PureFrameworkTensorFoundError


class KerasModel(KerasObject):
    """Add methods to this keras model object to have them added to every 
    tf.keras.models.Model object.

    This Keras object is simply a more convenient way to add custom functions to
    all TensorFlow Keras object types. When you add a function to this keras object,
    it will be added to EVERY native Keras layer type (i.e. tf.keras.models.Sequential)
    automatically by the TensorFlowHook.

    Note: all methods from AbstractObject will also be included because this
    object extends AbstractObject. So, if you're looking for a method on
    the native tf.keras API but it's not listed here, you might try
    checking AbstractObject.
    """

    def send(
        self, *location, inplace: bool = False, no_wrap=False, garbage_collect_data=True
    ):
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.

        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.
            inplace: if true,
              return the same object instance, else a new wrapper
            no_wrap: If True, wrap() is called on the created pointer
            garbage_collect_data: argument passed down to create_pointer()

        Returns:
            A tf.keras.layers.Layer[ObjectPointer] pointer to self. Note that this
            object will likely be wrapped by a ttf.keras.layers.Layer wrapper.
        """

        # If you send a pointer p1, you want the pointer to pointer p2
        # to control the garbage collection and not the remaining old
        # p1 (here self). Because if p2 is not GCed, GCing p1 shouldn't delete
        # the remote keras object, but if you want to do so, as p2 is not GCed,
        # you can still do `del p2`. This allows to chain multiple
        # .send().send() calls.

        if len(location) == 1:

            location = location[0]

            ptr = self.owner.send(
                self, location, garbage_collect_data=garbage_collect_data
            )

            ptr.description = self.description
            ptr.tags = self.tags

            # The last pointer should control remote GC,
            # not the previous self.ptr
            if hasattr(self, "ptr") and self.ptr is not None:
                ptr_ = self.ptr()
                if ptr_ is not None:
                    ptr_.garbage_collect_data = False

            # we need to cache this weak reference to the pointer so that
            # if this method gets called multiple times we can simply re-use
            # the same pointer which was previously created
            self.ptr = weakref.ref(ptr)

            if inplace:
                self.child = ptr
                return self
            else:
                output = ptr if no_wrap else ptr.wrap(type=tf.keras.models.Model)

        else:

            # TODO [Yann] check if we would want to send the keras
            # object to several workers this way.
            children = list()
            for loc in location:
                children.append(self.send(loc, no_wrap=True))

            output = syft.MultiPointerTensor(children=children)

            if not no_wrap:
                output = output.wrap(type=tf.keras.models.Model)

        return output
