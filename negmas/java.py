"""
Implements Java interoperability allowing parts of negmas to work smoothly with their Java counterparts in jnegmas

"""
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterable
import numpy as np

import pkg_resources
from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, JavaObject
# @todo use launch_gateway to start the java side. Will need to know the the jar location so jnegmas shoud save that
#  somewhere
from py4j.protocol import Py4JNetworkError

from negmas.helpers import get_class, instantiate, camel_case, snake_case

__all__ = [
    'JavaCallerMixin',
    'JNegmasGateway',
    'to_java',
    'from_java',
    'jnegmas_bridge_is_running',
    'init_jnegmas_bridge',
    'jnegmas_connection',
    'from_java',
]

DEFAULT_JNEGMAS_PATH = 'external/jnegmas-1.0-SNAPSHOT-all.jar'
PYTHON_CLASS_IDENTIFIER = '__python_class__'


@contextmanager
def jnegmas_connection(init: bool = False, path: Optional[str] = None, java_port=0
                       , python_port=0, client_server=True, shutdown=True):
    """A connection to jnegmas that closes automatically"""
    if init:
        JNegmasGateway.start_java_side(path=path, java_port=java_port)
    JNegmasGateway.connect(java_port=java_port, python_port=python_port, client_server=client_server)
    yield JNegmasGateway.gateway
    if shutdown:
        JNegmasGateway.shutdown()


def init_jnegmas_bridge(path: Optional[str] = None, port: int = 0):
    JNegmasGateway.start_java_side(path=path, java_port=port)


def jnegmas_bridge_is_running(port: int = None) -> bool:
    """
    Checks whether a JNegMAS Bridge is running. This bridge is needed to use any objects in the jnegmas package

    Remarks:

        You can start a JNegMAS Bridge in at least two ways:

        - execute the python function `init_jnegmas_bridge()` in this module
        - run "negmas jnegmas" on the terminal

    """
    return JNegmasGateway.is_running(port)


class PyEntryPoint:
    """Used as an entry point from the java side allowing it to create objects in python and take initiative in calling
    python"""

    def create(self, class_name: str, params: Dict[str, Any]) -> Any:
        """
        Creates a python object and returns it.

        Args:
            class_name: python full class name
            params: params to pass by value to the constructor of the object

        Remarks:
            - Notice that the returned object will only be usable in Java if it is implementing a Java interface which
              means that it *must* have an internal Java class with an `implements` list of interfaces that it
              implements.

        """
        params = {k: from_java(v) for k, v in params.items()}
        return instantiate(class_name, **params)

    class Java:
        implements = ['jnegmas.EntryPoint']


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
            path = pkg_resources.resource_filename('negmas', resource_name=DEFAULT_JNEGMAS_PATH)
        java_port = java_port if java_port > 0 else cls.DEFAULT_JAVA_PORT
        if jnegmas_bridge_is_running(port=java_port):
            return
        path = os.path.abspath(os.path.expanduser(path))
        try:
            subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
                f'java -jar {path} --doe --client-server --port {java_port}'
                , shell=True)
        except FileNotFoundError:
            print(os.getcwd(), flush=True)
            raise FileNotFoundError([os.getcwd(), path])
        except:
            pass
        time.sleep(0.5)
        cls.connect(auto_load_java=False)

    @classmethod
    def connect(cls, java_port: int = None, python_port: int = None
                , auto_load_java: bool = False, client_server: bool = True) -> bool:
        """
        connects to jnegmas
        """
        if not java_port:
            java_port = cls.DEFAULT_JAVA_PORT
        if not python_port:
            python_port = cls.DEFAULT_PYTHON_PORT
        if auto_load_java:
            if cls.gateway is None:
                cls.start_java_side(java_port=java_port)
            return True
        if cls.gateway is None:
            eager_load, auto_convert = True, True
            if client_server:
                cls.gateway = ClientServer(java_parameters=JavaParameters(port=java_port
                                                                          , auto_convert=auto_convert
                                                                          , eager_load=eager_load, auto_close=True
                                                                          , auto_gc=False, auto_field=False
                                                                          , daemonize_memory_management=True),
                                           python_parameters=PythonParameters(port=python_port
                                                                              , propagate_java_exceptions=True
                                                                              , daemonize=True
                                                                              , eager_load=eager_load
                                                                              , auto_gc=False
                                                                              , daemonize_connections=True
                                                                              ),
                                           python_server_entry_point=PyEntryPoint())
            else:
                pyparams = CallbackServerParameters(port=python_port
                                                    , daemonize_connections=True
                                                    , daemonize=True
                                                    , eager_load=True
                                                    , propagate_java_exceptions=True)
                cls.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=java_port
                                                                               , auto_convert=auto_convert
                                                                               , auto_field=False
                                                                               , eager_load=eager_load
                                                                               , auto_close=True),
                                          callback_server_parameters=pyparams, auto_convert=auto_convert
                                          , start_callback_server=True, eager_load=eager_load
                                          , python_server_entry_point=PyEntryPoint())
                python_port = cls.gateway.get_callback_server().get_listening_port()
                cls.gateway.java_gateway_server.resetCallbackClient(
                    cls.gateway.java_gateway_server.getCallbackClient().getAddress(),
                    python_port)
        return True

    @classmethod
    def shutdown(cls):
        cls.gateway.shutdown(raise_exception=False)
        cls.gateway.shutdown_callback_server(raise_exception=False)

    @classmethod
    def is_running(cls, port):
        """
        Checks whether a JNegMAS Bridge is running. This bridge is needed to use any objects in the jnegmas package

        Remarks:

            You can start a JNegMAS Bridge in at least two ways:

            - execute the python function `init_jnegmas_bridge()` in this module
            - run "negmas jnegmas" on the terminal

        """
        if port is None:
            port = JNegmasGateway.DEFAULT_JAVA_PORT
        s = socket.socket()
        try:
            s.connect(('127.0.0.1', port))
            return True
        except ConnectionRefusedError:
            return False
        except IndexError:
            return False
        except Py4JNetworkError:
            return False
        except Exception:
            return False
        finally:
            s.close()


def java_identifier(s: str):
    if s != PYTHON_CLASS_IDENTIFIER:
        return camel_case(s)
    return s


def to_dict_for_java(value, deep=True, add_type_field=True):
    """Encodes the given value as nothing not more complex than simple dict of either dicts, lists or builtin numeric
    or string values

    Args:
        value: Any object
        deep: Whether we should go deep in the encoding or do a shallow encoding
        add_type_field: Whether to add a type field. If True, A field named `PYTHON_CLASS_IDENTIFIER` will be added
        giving the type of `value`

    Remarks:

        - All iterables are converted to lists when `deep` is true.
        - If the `value` object has a `to_java` member, it will be called to do the conversion, otherwise its `__dict__`
          or `__slots__` member will be used.

    See Also:
          `from_java`, `PYTHON_CLASS_IDENTIFIER`

    """
    def good_field(k: str):
        return not k.startswith('python_') and not k.startswith('java') and not (k != PYTHON_CLASS_IDENTIFIER and k.startswith('_'))
    if isinstance(value, JavaObject):
        return value
    if isinstance(value, dict):
        if not deep:
            return {java_identifier(k): v for k, v in value.items()}
        return {java_identifier(k): to_dict_for_java(v) for k, v in value.items() if good_field(k)}
    if isinstance(value, Iterable) and not deep:
        return value
    if isinstance(value, Iterable) and not isinstance(value, str):
        return [to_dict_for_java(_) for _ in value]
    if hasattr(value, 'to_java'):
        converted = value.to_java()
        if isinstance(converted, dict):
            if add_type_field:
                converted[PYTHON_CLASS_IDENTIFIER] = value.__class__.__module__ + '.' + value.__class__.__name__
            return {java_identifier(k): v for k, v in converted.items()}
        else:
            return converted
    if hasattr(value, '__dict__'):
        if deep:
            d = {java_identifier(k): to_dict_for_java(v) for k, v in value.__dict__.items()
                 if good_field(k)}
        else:
            d = {java_identifier(k): v for k, v in value.__dict__.items()
                 if good_field(k)}
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = value.__class__.__module__ + '.' + value.__class__.__name__
        return d
    if hasattr(value, '__slots__'):
        if deep:
            d = dict(zip((camel_case(k) for k in value.__slots__), (to_dict_for_java(getattr(value, _)) for _ in value.__slots__)))
        else:
            d = dict(zip((camel_case(k) for k in value.__slots__), (getattr(value, _) for _ in value.__slots__)))
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = value.__class__.__module__ + '.' + value.__class__.__name__
        return d
    # a builtin
    if isinstance(value, np.int64):
        return int(value)
    return value


def to_java(value):
    """Encodes the given value as nothing not more complex than simple dict of either dicts, lists or builtin numeric
    or string values

    Args:
        value: Any object
        deep: Whether we should go deep in the encoding or do a shallow encoding
        add_type_field: Whether to add a type field. If True, A field named `PYTHON_CLASS_IDENTIFIER` will be added
        giving the type of `value`

    Remarks:

        - All iterables are converted to lists when `deep` is true.
        - If the `value` object has a `to_java` member, it will be called to do the conversion, otherwise its `__dict__`
          or `__slots__` member will be used.

    See Also:
          `from_java`, `PYTHON_CLASS_IDENTIFIER`

    """
    value = to_dict_for_java(value, deep=True, add_type_field=True)
    if not isinstance(value, dict):
        return value
    # print('')
    for k, v in value.items():
        # if v is not None and not isinstance(v, int) and not isinstance(v, float) and not isinstance(v, str) \
        #     and not isinstance(v, list) and not isinstance(v, dict) and not isinstance(v, tuple):
        #     print(f'\t{k}: {v}({type(v)})')
        if isinstance(v, np.int64):
            value[k] = int(v)
    return JNegmasGateway.gateway.entry_point.createJavaObjectFromMap(value)


def from_java(d: Dict[str, Any], deep=True, remove_type_field=True, fallback_class_name: Optional[str]=None):
    """Decodes a dict coming from java recovering all objects in the way

    Args:
        d: The value to be decoded. If it is not a dict, it is returned as it is.
        deep: If true, decode recursively
        remove_type_field: If true the field called `PYTHON_CLASS_IDENTIFIER` will be removed if found.
        fallback_class_name: If given, it is used as the fall-back  type if ``PYTHON_CLASS_IDENTIFIER` is not in the dict.

    Remarks:

        - If the object is not a dict or if it has no `PYTHON_CLASS_IDENTIFIER` field and no `fallback_class_name` is
          given, the input `d` is returned as it is. It will not even be copied.

    See Also:
        `to_java`, `PYTHON_CLASS_IDENTIFIER`



    """
    def good_field(k: str):
        return not k.startswith('python_') and not k.startswith('java') and not (k != PYTHON_CLASS_IDENTIFIER and k.startswith('_'))
    if not isinstance(d, dict):
        return d
    if remove_type_field:
        python_class_name = d.pop(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
    else:
        python_class_name = d.get(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
    if python_class_name is not None:
        python_class = get_class(python_class_name)
        # we resolve sub-objects first from the dict if deep is specified before calling from_java on the class
        if deep:
            d = {snake_case(k): from_java(v) for k, v in d.items() if good_field(k)}
        # from_java needs to do a shallow conversion from a dict as deep conversion is taken care of already.
        if hasattr(python_class, 'from_java'):
            return python_class.from_java({snake_case(k): v for k, v in d.items()})
        return python_class({snake_case(k): v for k, v in d.items() if good_field(k)})
    return d


class JavaCallerMixin:
    """A mixin to enable calling a java counterpart. This mixin can ONLY be used with a `NamedObject` because it uses
    its id property.

    Other than inheriting this mixin, you should call its `init_java_bridge` in your `__init__` (or whenever your object
    is initialized and you need to create the Java counterpart). You should then implement all your functions as calls
    to `java_object`.

    If for example you have a function `do_this`, you can just define it as follows:

    .. code

        def do_this(self, x, y, z) -> Type:
            return self.java_object.doThis(x, y, z)

    Notice that you cannot use named arguments when calling the function in `java_object` and that the names are
    converted to camelCase instead of snake_case. Moreover, property x will be implemented as a pair getX, setX on the
    Java side.

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
    def from_dict(cls, java_object, *args, **kwargs):
        """Creates a Python object representing the corresponding Java object"""
        obj = cls(*args, **kwargs)
        obj.java_object = java_object
        obj._connected = True
        obj.java_class_name = java_object.getClass().getSimpleName()
        return obj

    def init_java_bridge(self, java_class_name: str, auto_load_java: bool = False
                         , python_shadow_object: Any = None):
        """
        initializes a connection to the java bridge creating a member called `java_object` that can be used to access
        the counterpart object in Java

        Args:
            java_class_name: The type of the Java object to be created
            auto_load_java: When true, a JVM will be automatically created (if one is not available)
            python_shadow_object: A python object to shadow the java object. The object will just call the corresponding
            method on this shadow object whenever it needs.


        Remarks:

            - sets a member called java_object that can be used to access the corresponding Java object crated
            - if `python_shadow_object` is given, it must be an object of a type that has an internal class called
              Java which has a single member called 'implements' which is a list of one string element
              representing the Java interface being implemented (it must be either jnegmas.PyCallable or an extension of
              it).
        """
        self.java_class_name = java_class_name
        self._connected = JNegmasGateway.connect(auto_load_java=auto_load_java)
        self.java_object = self._create(python_shadow_object)

    def _create(self, python_shadow: Any = None):
        """
        Creates the internal object.

        Args:

              python_shadow: An object that has an internal Java class with implements = ['jnegmas.Callable']

        """
        if python_shadow is None:
            return JNegmasGateway.gateway.entry_point.create(self.java_class_name)
        return JNegmasGateway.gateway.entry_point.create(self.java_class_name, python_shadow)
