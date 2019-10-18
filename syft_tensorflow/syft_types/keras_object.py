import weakref

import tensorflow as tf

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.object import AbstractObject
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker

from syft.exceptions import PureFrameworkTensorFoundError


class KerasObject(AbstractObject):
    """Add methods to this keras object to have them added for example added to every 
    tf.keras.layers.Layer object.

    This Keras object is simply a more convenient way to add custom functions to
    all TensorFlow Keras object types. When you add a function to this keras object,
    it will be added to EVERY native Keras layer type (i.e. tf.keras.layers.Dense)
    automatically by the TensorFlowHook.

    Note: all methods from AbstractObject will also be included because this
    object extends AbstractObject. So, if you're looking for a method on
    the native tf.keras API but it's not listed here, you might try
    checking AbstractObject.
    """

    def has_child(self):
        return hasattr(self, "child")
        
    def describe(self, description):
        self.description = description
        return self

    def tag(self, *_tags):
        if self.tags is None:
            tags = list()
        else:
            tags = list(self.tags)

        for new_tag in _tags:
            tags.append(new_tag)

        self.tags = set(tags)
        return self

    @property
    def tags(self):
        if self.has_child():
            return self.child.tags
        else:
            if not hasattr(self, "_tags"):
                self._tags = None
            return self._tags

    @tags.setter
    def tags(self, new_tags):
        if self.has_child():
            if new_tags is not None:
                self.child.tags = set(new_tags)
            else:
                self.child.tags = set()
        else:
            self._tags = new_tags

    @property
    def description(self):
        if self.has_child():
            return self.child.description
        else:
            if not hasattr(self, "_description"):
                self._description = None
            return self._description

    @description.setter
    def description(self, new_desc):
        if self.has_child():
            self.child.description = new_desc
        else:
            self._description = new_desc

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
                output = ptr if no_wrap else ptr.wrap(type=tf.keras.layers.Layer)

        else:

            # TODO [Yann] check if we would want to send the keras
            # object to several workers this way.
            children = list()
            for loc in location:
                children.append(self.send(loc, no_wrap=True))

            output = syft.MultiPointerTensor(children=children)

            if not no_wrap:
                output = output.wrap(type=tf.keras.layers.Layer)

        return output

    def get(self, *args, inplace: bool = False, **kwargs):
        """Requests the Keras object being pointed to, be serialized and return
            Args:
                args: args to forward to worker
                inplace: if true, return the same object instance,
                  else a new wrapper
                kwargs: kwargs to forward to worker
            Raises:
                GetNotPermittedError: Raised if
                  get is not permitted on this Keras object
        """
        # Transfer the get() to the child attribute which is a pointer

        obj = self.child.get(*args, **kwargs)

        if inplace:
            return self
        else:
            return obj

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ) -> ObjectPointer:
        """Creates a pointer to the "self" tf.keras.layers.Layer object.

        Returns:
            A ObjectPointer pointer to self. Note that this
            object will likely be wrapped by a tf.keras.layers.Layer wrapper.
        """
        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location is not None and location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = syft.ID_PROVIDER.pop()

        ptr = ObjectPointer.create_pointer(
            self,
            location,
            id_at_location,
            register,
            owner,
            ptr_id,
            garbage_collect_data,
        )

        return ptr

    def __str__(self) -> str:
        if self.has_child():
            if self.is_wrapper:
                return "(Wrapper)>" + self.child.__str__()
            else:
                return type(self).__name__ + ">" + self.child.__str__()
        else:
            return self.native___str__()

    def __repr__(self) -> str:
        if self.has_child():
            if self.is_wrapper:
                return "(Wrapper)>" + self.child.__str__()
            else:
                return type(self).__name__ + ">" + self.child.__repr__()
        else:
            out = self.native___repr__()

        big_repr = False

        if self.tags is not None and len(self.tags):
            big_repr = True
            out += "\n\tTags: "
            for tag in self.tags:
                out += str(tag) + " "

        if self.description is not None:
            big_repr = True
            out += (
                "\n\tDescription: " + str(self.description).split("\n")[0] + "..."
            )

        return out

    @classmethod
    def handle_func_command(cls, command):
        """
        Instantiate the native Keras object.

        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        cmd, _, args, kwargs = command

        # TODO: clean this line
        cmd_split = cmd.split(".")
        cmd_path = cmd_split[:-1]
        cmd_name = cmd_split[-1]
        cmd = "syft.local_worker.hook." + ".".join(cmd_path) + ".native_" + cmd_name

        # Run the native function with the new args
        # Note the the cmd should already be checked upon reception by the worker
        # in the execute_command function
        if isinstance(args, tuple):
            response = eval(cmd)(*args, **kwargs)
        else:
            response = eval(cmd)(args, **kwargs)

        return response
