import weakref

import tensorflow as tf

import syft
from syft.generic.tensor import AbstractTensor
from syft.workers.base import BaseWorker
from syft.generic.pointers.pointer_tensor import PointerTensor

from syft.exceptions import PureFrameworkTensorFoundError
from syft.generic.frameworks.types import FrameworkTensor

from syft.generic.frameworks.hook import hook_args


class TensorFlowTensor(AbstractTensor):
    """Add methods to this tensor to have them added to every tf.Tensor object.

    This tensor is simply a more convenient way to add custom functions to
    all TensorFlow tensor types. When you add a function to this tensor,
    it will be added to EVERY native TF tensor type (i.e. tf.Tensor)
    automatically by the TensorFlowHook.

    Note: all methods from AbstractTensor will also be included because this
    tensor extends AbstractTensor. So, if you're looking for a method on
    the native TensorFlow tensor API but it's not listed here, you might try
    checking AbstractTensor.
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

    @property
    def shape(self):
        if self.is_wrapper:
            return self.child.shape
        else:
            return self.native_shape

    def send(
        self,
        *location,
        inplace: bool = False,
        # local_autograd=False,
        # preinitialize_grad=False,
        no_wrap=False,
        garbage_collect_data=True,
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
            # local_autograd: Use autograd system on the local machine instead
              of TensorFlow's autograd on the workers.
            # preinitialize_grad: Initialize gradient for AutogradTensors
              to a tensor
            no_wrap: If True, wrap() is called on the created pointer
            garbage_collect_data: argument passed down to create_pointer()

        Returns:
            A tf.EagerTensor[PointerTensor] pointer to self. Note that this
            object will likely be wrapped by a tf.EagerTensor wrapper.
        """

        # If you send a pointer p1, you want the pointer to pointer p2
        # to control the garbage collection and not the remaining old
        # p1 (here self). Because if p2 is not GCed, GCing p1 shouldn't delete
        # the remote tensor, but if you want to do so, as p2 is not GCed,
        # you can still do `del p2`. This allows to chain multiple
        # .send().send() calls.

        if len(location) == 1:

            location = location[0]

            if hasattr(self, "child") and isinstance(
                self.child, PointerTensor
            ):
                self.child.garbage_collect_data = False

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
                self.set_()  # TODO[jason]: pretty sure this is torch specific
                self.child = ptr
                return self
            else:
                output = ptr if no_wrap else ptr.wrap(type=tf.constant, value=[])

        else:

            children = list()
            for loc in location:
                children.append(self.send(loc, no_wrap=True))

            output = syft.MultiPointerTensor(children=children)

            if not no_wrap:
                output = output.wrap(type=tf.constant, value=[])

        return output

    def get(self, *args, inplace: bool = False, **kwargs):
        """Requests the tensor/chain being pointed to, be serialized and return
            Args:
                args: args to forward to worker
                inplace: if true, return the same object instance,
                  else a new wrapper
                kwargs: kwargs to forward to worker
            Raises:
                GetNotPermittedError: Raised if
                  get is not permitted on this tensor
        """
        # Transfer the get() to the child attribute which is a pointer

        tensor = self.child.get(*args, **kwargs)

        # Clean the wrapper
        delattr(self, "child")

        if inplace:
            self.set_(tensor)  # TODO[jvmancuso]: torch-specific
            if hasattr(tensor, "child"):
                self.child = tensor.child
            return self
        else:
            return tensor

    def create_pointer(
        self,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
        shape=None,
    ) -> PointerTensor:
        """Creates a pointer to the "self" tf.Tensor object.

        Returns:
            A PointerTensor pointer to self. Note that this
            object will likely be wrapped by a tf.Tensor wrapper.
        """
        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location is not None and location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = syft.ID_PROVIDER.pop()

        if shape is None:
            shape = self.shape

        ptr = syft.PointerTensor.create_pointer(
            self,
            location,
            id_at_location,
            register,
            owner,
            ptr_id,
            garbage_collect_data,
            shape,
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

            if big_repr:
                out += "\n\tShape: " + str(self.shape.as_list())

            return out

    @classmethod
    def handle_func_command(cls, command):
        """
        Operates as a router for functions. A function call always starts
        by being handled here and 3 scenarii must be considered:

        Real TensorFlow tensor:
            The arguments of the function are real tensors so we should
            run the native TensorFlow command

        TensorFlow wrapper:
            The arguments are just wrappers at the top of a chain
            (ex: wrapper>LoggingTensor>TensorFlow tensor), so just forward
            the instruction to the next layer type in the chain (in
            the example above to LoggingTensor.handle_func_command),
            get the response and replace a wrapper on top of all tensors
            found in the response.

        Syft Tensor:
            The arguments are syft tensors of same type: this can happen
            if at any node of the chain where some function is forwarded,
            the handle_func_command modify the function and make a new
            call but keeps the arguments "un-wrapped". Making a new call
            means that by default the command is treated here in the
            global router.

        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        cmd, _, args, kwargs = command

        try:  # will work if tensors are wrappers

            # Replace all TensorFlow tensor with their child attribute
            # Note that we return also args_type which helps handling case 3 in the docstring
            new_args, new_kwargs, new_type, args_type = hook_args.unwrap_args_from_function(
                cmd, args, kwargs, return_args_type=True
            )
            # This handles case 3: it redirects the command to the appropriate class depending
            # of the syft type of the arguments and returns
            if args_type not in FrameworkTensor:
                return args_type.handle_func_command(command)

            # build the new command
            new_command = (cmd, None, new_args, new_kwargs)
            # Send it to the appropriate class and get the response
            response = new_type.handle_func_command(new_command)
            # Put back the wrappers where needed
            response = hook_args.hook_response(cmd, response, wrap_type=args_type)
        except PureFrameworkTensorFoundError:  # means that it's not a wrapper but a pure tensor

            # Check that the function has not been overwritten
            try:
                # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
                command = cls.rgetattr(cls, cmd)
                return command(*args, **kwargs)
            except AttributeError:
                pass

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
