"""
Implements Java interoperability allowing parts of negmas to work smoothly with their Java counterparts in jnegmas

"""
import os
import subprocess
import time
from typing import Optional, Dict, Any, Iterable, Callable

import pkg_resources
from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from py4j.java_collections import MapConverter, SetConverter, ListConverter
from py4j.java_gateway import JavaClass

# @todo use launch_gateway to start the java side. Will need to know the the jar location so jnegmas shoud save that
#  somewhere

__all__ = [
    'JavaCallerMixin',
    'JNegmasGateway',
    'JavaConvertible',
    'to_java',
]


def dict_encode(value):
    """Encodes the given value as nothing not more complex than simple dict of either dicts, lists or builtin numeric
    or string values"""
    if isinstance(value, dict):
        return {k: dict_encode(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, str):
        return [dict_encode(_) for _ in value]
    return value


def to_java(object) -> Dict[str, Any]:
    """Converts the object into a dict for serialization to Java side
    """
    d = dict_encode(object.__dict__)
    # d['__java_class_name__'] = object.__class__.__name__
    # if hasattr(object, 'Java') and isinstance(object.Java, type):
    #     d['python_object'] = object
    # else:
    #     d['python_object'] = None
    return d


class JavaConvertible:
    """Object that represent *readonly* data that can be sent to jnegmas."""

    def to_java(self) -> Dict[str, Any]:
        return to_java(self)


class _NegmasConverter:
    """An internal  class for converting `JavaConvertible` objects"""

    def can_convert(self, x):
        return isinstance(x, JavaConvertible)

    def convert(self, x, gateway_client):
        """We assume that a class with the same name exists in jnegmas"""
        class_name = f'j' + x.__class__.__module__ + '.' + x.__class__.__name__
        java_object = JavaClass(class_name, gateway_client)()
        java_object.fill(x.to_java())
        return java_object


# registering the Negmas converter globally in py4j
# register_input_converter(converter=_NegmasConverter(), prepend=True)


class JNegmasGateway:
    DEFAULT_PYTHON_PORT = 25334
    DEFAULT_JAVA_PORT = 25333

    gateway: Optional[ClientServer] = None

    @classmethod
    def start_java_side(cls, path: str = None, java_port: int = 0) -> None:
        """Initializes a connection to jnegmas

        Examples:

            # >>> start_java_side(port=35337)
            # >>> a = JNegmasGateway.do_nothing_manager()
            # >>> a.java_uuid.startswith('jnegmas')
            # True
            # >>> len(a.java_uuid)- len(a.java_class_name) == 36 # length of UUID
            # True

        """
        if cls.gateway is not None:
            return
        if path is None:
            path = pkg_resources.resource_filename('negmas', resource_name='external/jnegmas.jar')
        java_port = java_port if java_port > 0 else cls.DEFAULT_JAVA_PORT
        path = os.path.abspath(os.path.expanduser(path))
        try:
            subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
                f'java -jar {path} --doe --port {java_port}'
                , shell=True)
        except FileNotFoundError:
            print(os.getcwd(), flush=True)
            raise FileNotFoundError([os.getcwd(), path])
        except:
            pass
        time.sleep(0.5)
        cls.connect(auto_load_java=False)

    @classmethod
    def connect(cls, auto_load_java: bool = False) -> bool:
        """
        connects to jnegmas
        """
        if auto_load_java:
            if cls.gateway is None:
                cls.start_java_side()
            return True
        if cls.gateway is None:
            cls.gateway = ClientServer(java_parameters=JavaParameters(port=JNegmasGateway.DEFAULT_JAVA_PORT
                                                                      , auto_convert=True),
                                       python_parameters=PythonParameters(port=JNegmasGateway.DEFAULT_PYTHON_PORT
                                                                          , propagate_java_exceptions=True))
        return True


class JavaCallerMixin:
    """A mixin to enable calling a java counterpart. This mixin can ONLY be used with a `NamedObject` because it uses
    its id property.

    Other than inheriting this mixin, you should call its `init_java_bridge` in your `__init__` (or whenever your object
    is initialized and you need to create the Java counterpart). You should then implement all your functions as calls
    to `java_object`.

    If for example you have a function `do_this`, you can just define it as follows:

    .. code

        def do_this(self, x, y, z) -> Type:
            return self.java_object.do_this(x, y, z)

    Notice that you cannot use named arguments when calling the function in `java_object`

    If your class needs just to call the corresponding java object but is never called back from it then you are done
    after inheriting from this mixin.

    If your objects need to be called from the java counterpart, then you need to add the following to your class
    definition

    .. code

        class Java:
            implements = ['jnegmas.ClassName']

    This assumes that your class is named `ClassName` and that there is an interface called `ClassName` defined in
    jnegmas that has the *same* public interface as your class (or whatever part of it to be called from Java).

    """

    @classmethod
    def from_java(cls, java_object, *args, **kwargs):
        """Creates a Python object representing the corresponding Java object"""
        obj = cls(*args, **kwargs)
        obj.java_object = java_object
        obj._connected = True
        obj._java_class_name = java_object.getClass().getSimpleName()
        return obj

    def init_java_bridge(self, java_class_name: str, auto_load_java: bool = False
                         , python_object_factory: Optional[Callable[[], Any]] = None):
        """
        initializes a connection to the java bridge creating a member called `java_object` that can be used to access
        the counterpart object in Java

        Args:
            java_class_name: The type of the Java object to be created
            auto_load_java: When true, a JVM will be automatically created (if one is not available)
            python_object_factory: A callable that creates a python object to shadow the java object


        Remarks:

            - sets a member called java_object that can be used to access the corresponding Java object crated
            - if `python_object_factory` is given, it must return an object of a type that has an internal class called
              Java which has a single member called 'implements' which is a list of one string element
              representing the Java interface being implemented (it must be either jnegmas.PyCallable or an extension of
              it).
        """
        self._java_class_name = java_class_name
        self._connected = JNegmasGateway.connect(auto_load_java=auto_load_java)
        self.java_object = self._create(python_object_factory() if python_object_factory else None)

    def _create(self, python_shadow: Any = None):
        """
        Creates the internal object.

        Args:

              python_shadow: An object that has an internal Java class with implements = ['jnegmas.PyCallable']

        """
        if python_shadow is None:
            return JNegmasGateway.gateway.entry_point.create(self._java_class_name)
        return JNegmasGateway.gateway.entry_point.create(self._java_class_name, python_shadow)


