from types import ModuleType
import typing

from tensorflow.python.framework.ops import EagerTensor
from syft.generic.frameworks.attributes import FrameworkAttributes

from syft_tensorflow.tensor import TensorFlowTensor

if typing.TYPE_CHECKING:
    from syft_tensorflow.hook import TensorFlowHook


class TensorFlowAttributes(FrameworkAttributes):
    """Adds tensorflow module related custom attributes.

    TensorFlowAttributes is a special class where all custom attributes related
    to the torch module can be added. Any global parameter, configuration,
    or reference relating to TensorFlow should be stored here instead of
    attaching it directly to some other part of the global namespace.

    The main reason we need this is because the hooking process occasionally
    needs to save global objects, notably including what methods to hook and
    what methods to NOT hook.

    This will hold all necessary attributes PySyft needs.

    Args:
        tensorflow: A ModuleType indicating the tensorflow module
        hook: A TensorFlowHook to stash
    """

    ALIAS = "tensorflow"
    Tensor = TensorFlowTensor

    def __init__(self, tensorflow: ModuleType, hook: ModuleType):
        super().__init__(tensorflow, hook)
        # Stash the hook here for global access elsewhere
        self.hook = hook

        # List modules that we will hook
        self.tensorflow_modules = {
            "tensorflow": tensorflow,
            "tensorflow.keras.activations": tensorflow.keras.activations,
            "tensorflow.math": tensorflow.math,
            # "tensorflow.keras": tensorflow.keras,
        }

        # Set of all function names with module as prefix in the modules to hook
        self._tensorflow_modules_functions = {
            f"{module_name}.{func_name}"
            for module_name, tensorflow_module
            in self.tensorflow_modules.items()

            for func_name in dir(tensorflow_module)
        }

        # Store reference to all tf functions by string name
        # stored in tensorflow_modules_functions
        self.eval_tensorflow_modules_functions = {
            f"{module_name}.{func_name}": getattr(tensorflow_module, func_name)
            for module_name, tensorflow_module
            in self.tensorflow_modules.items()

            for func_name in dir(tensorflow_module)
        }

        # Add special functions to exclude from the hook
        # **in alphabetical order**
        # Reasons can be:
        # - Used for internal process like printing tensors
        # - Don't use tensors so are bound to have local executions
        # - etc
        # DON'T put here:
        # - functions like native_*
        # - functions that could use pointers or syft tensors
        self.exclude = []

        # SECTION: List all torch tensor methods we want to overload
        self.tensor_types = [tensorflow.Tensor, tensorflow.Variable]

        # SECTION: Build the guard, that define which
        # functions or methods can be safely called by
        # external or local workers

        # Add all tensor types
        self.guard = {
            "Tensor": tensorflow.Tensor,
            "Variable": tensorflow.Variable,
            "EagerTensor": EagerTensor,
        }

        # Allow the `syft.` prefix to be used
        keys = list(self.guard.keys())
        for key in keys:
            self.guard[f"syft.{key}"] = self.guard[key]

        # Concatenate torch functions and torch methods
        self.allowed_commands = self._tensorflow_modules_functions

        # The equivalent concatenation of native torch function
        # names and native torch method names
        self.native_commands = {
            command_name: self.get_native_framework_name(command_name)
            for command_name in self.allowed_commands
        }

        self.command_guard = self._command_guard

        self.exclude = [
            '__getattr__',
            '__finalize__',
            '_constructor',
            '_init_mgr',
        ]

        self.inplace_methods = {}


    def is_inplace_method(self, method):
        # I've not yet encountered any inplace methods in TF
        return False
