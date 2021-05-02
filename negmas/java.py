"""
Implements Java interoperability allowing parts of negmas to work smoothly
with their Java counterparts in jnegmas

"""
import os
import socket
import subprocess
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pkg_resources
from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from py4j.java_collections import JavaList, JavaMap, JavaSet, ListConverter
from py4j.java_gateway import (
    CallbackServerParameters,
    GatewayParameters,
    JavaGateway,
    JavaObject,
)

from py4j.protocol import Py4JNetworkError

from .helpers import camel_case, get_class, snake_case
from .serialization import serialize

__all__ = [
    "JavaCallerMixin",
    "JNegmasGateway",
    "to_java",
    "from_java",
    "java_identifier",
    "jnegmas_bridge_is_running",
    "init_jnegmas_bridge",
    "jnegmas_connection",
    "from_java",
    "to_dict",
    # "from_dict",
    "java_link",
    "PYTHON_CLASS_IDENTIFIER",
]

DEFAULT_JNEGMAS_PATH = "external/jnegmas-1.0-SNAPSHOT-all.jar"
PYTHON_CLASS_IDENTIFIER = "__python_class__"


@contextmanager
def jnegmas_connection(
    init: bool = False,
    path: Optional[str] = None,
    java_port=0,
    python_port=0,
    client_server=True,
    shutdown=True,
):
    """A connection to jnegmas that closes automatically"""
    if init:
        JNegmasGateway.start_java_side(path=path, java_port=java_port)
    JNegmasGateway.connect(
        java_port=java_port, python_port=python_port, client_server=client_server
    )
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
        return from_java(params, fallback_class_name=class_name)

    def create_shadow(self, class_name: str, params: Dict[str, Any]) -> Any:
        lst = class_name.split(".")
        for ending in ("Negotiator", "FactoryManager", "UtilityFunction"):
            if lst[-1].endswith(ending):
                lst[-1] = "_Shadow" + (
                    ending if ending != "Negotiator" else "SAONegotiator"
                )
                break
        else:
            lst[-1] = "_Shadow" + lst[-1]
        shadow_class_name = ".".join(lst)
        cls = get_class(shadow_class_name)
        obj = from_java(params, fallback_class_name=class_name)
        shadow = cls(obj)
        return shadow

    class Java:
        implements = ["jnegmas.EntryPoint"]


class JNegmasGateway:
    DEFAULT_PYTHON_PORT = 25334
    DEFAULT_JAVA_PORT = 25333

    gateway: Optional[Union[JavaGateway, ClientServer]] = None

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
            path = pkg_resources.resource_filename(
                "negmas", resource_name=DEFAULT_JNEGMAS_PATH
            )
        java_port = java_port if java_port > 0 else cls.DEFAULT_JAVA_PORT
        if jnegmas_bridge_is_running(port=java_port):
            return
        path = os.path.abspath(os.path.expanduser(path))
        try:
            subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
                f"java -jar {path} --doe --client-server --port {java_port}", shell=True
            )
        except FileNotFoundError:
            warnings.warn(os.getcwd(), flush=True)
            raise FileNotFoundError([os.getcwd(), path])
        except:
            pass
        time.sleep(0.5)
        cls.connect(auto_load_java=False)

    @classmethod
    def connect(
        cls,
        java_port: int = None,
        python_port: int = None,
        auto_load_java: bool = False,
        client_server: bool = True,
    ) -> bool:
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
            auto_field, auto_gc = True, False
            propagate_exceptions = False
            if client_server:
                cls.gateway = ClientServer(
                    java_parameters=JavaParameters(
                        port=java_port,
                        auto_convert=auto_convert,
                        eager_load=eager_load,
                        auto_close=True,
                        auto_gc=False,
                        auto_field=auto_field,
                        daemonize_memory_management=True,
                    ),
                    python_parameters=PythonParameters(
                        port=python_port,
                        propagate_java_exceptions=propagate_exceptions,
                        daemonize=True,
                        eager_load=eager_load,
                        auto_gc=auto_gc,
                        daemonize_connections=True,
                    ),
                    python_server_entry_point=PyEntryPoint(),
                )
            else:
                pyparams = CallbackServerParameters(
                    port=python_port,
                    daemonize_connections=True,
                    daemonize=True,
                    eager_load=eager_load,
                    propagate_java_exceptions=propagate_exceptions,
                )
                cls.gateway = JavaGateway(
                    gateway_parameters=GatewayParameters(
                        port=java_port,
                        auto_convert=auto_convert,
                        auto_field=auto_field,
                        eager_load=eager_load,
                        auto_close=True,
                    ),
                    callback_server_parameters=pyparams,
                    auto_convert=auto_convert,
                    start_callback_server=True,
                    eager_load=eager_load,
                    python_server_entry_point=PyEntryPoint(),
                )
            python_port = cls.gateway.get_callback_server().get_listening_port()
            cls.gateway.java_gateway_server.resetCallbackClient(
                cls.gateway.java_gateway_server.getCallbackClient().getAddress(),
                python_port,
            )
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
            s.connect(("127.0.0.1", port))
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


def java_link(obj, map=None):
    """
    Creates a link in java to the object given without copying it.

    Args:

        obj: The object for which to create a java shadow
        map: construction parameters
        copyable: If true, we will assume that the java object is PyCopyable otherwise PyConstructable. Only checked if
        map is not None

    Returns:
        A java object. Cannot be used directly in python but can be used as an argument to a call to of a java object.

    """
    class_name = obj.__class__.__module__ + "." + obj.__class__.__name__
    lst = class_name.split(".")
    lst[0] = "j" + lst[0]
    if lst[-1].startswith("_Shadow"):
        lst[-1] = lst[-1][len("_Shadow") :]
    if lst[-1].startswith("Shadow"):
        lst[-1] = lst[-1][len("Shadow") :]
    lst[-1] = "Python" + lst[-1]
    class_name = ".".join(lst)
    java_obj = JNegmasGateway.gateway.entry_point.create(class_name, obj)
    if map is not None:
        java_obj.fromMap(map)
    else:
        java_obj.fromMap(serialize(obj))
    return java_obj


def to_java(value, add_type_field=True, python_class_name: str = None):
    """Encodes the given value as nothing not more complex than simple dict
    of either dicts, lists or builtin numeric or string values

    Args:

        value: Any object
        add_type_field: If true, the `PYTHON_CLASS_IDENTIFIER`  will be added
                        with the python class field on it
        python_class_name: It given it overrides the class name written when
                           `add_type_field` is True otherwise, the class name
                           will be inferred as the __class__ of `value`.

    Remarks:

        - All iterables are converted to lists when `deep` is true.
        - If the `value` object has a `to_java` member, it will be called to do the conversion, otherwise its `__dict__`
          or `__slots__` member will be used.

    See Also:
          `from_java`, `PYTHON_CLASS_IDENTIFIER`

    """
    if value is None:
        return None
    value = serialize(value, deep=True, add_type_field=add_type_field)
    if isinstance(value, JavaObject):
        return value
    if isinstance(value, np.int64):
        return int(value)
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, np.int64):
                value[k] = int(v)
        if add_type_field and python_class_name is not None:
            value[PYTHON_CLASS_IDENTIFIER] = python_class_name
        return JNegmasGateway.gateway.entry_point.createJavaObjectFromMap(value)

    if isinstance(value, Iterable) and not isinstance(value, str):
        return ListConverter().convert(
            [
                JNegmasGateway.gateway.entry_point.createJavaObjectFromMap(_)
                if isinstance(_, dict)
                else _
                for _ in value
            ],
            JNegmasGateway.gateway._gateway_client,
        )
    return value


def python_identifier(k: str) -> str:
    """
    Converts a key to snake case keeping dunder keys alone

    Args:

        k: string to be converted

    Returns:

        snake-case string
    """
    return k if k.startswith("__") else snake_case(k)


def from_java(
    d: Any, deep=True, remove_type_field=True, fallback_class_name: Optional[str] = None
):
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
        return (
            not k.startswith("python_")
            and not k.startswith("java")
            and not (k != PYTHON_CLASS_IDENTIFIER and k.startswith("_"))
        )

    if d is None or isinstance(d, int) or isinstance(d, float) or isinstance(d, str):
        return d
    if isinstance(d, JavaList):
        return [_ for _ in d]
    if isinstance(d, JavaSet):
        return {_ for _ in d}
    if isinstance(d, JavaMap):
        if deep:
            d = {python_identifier(k): from_java(v) for k, v in d.items()}
        else:
            d = {python_identifier(k): v for k, v in d.items()}
        deep = False
    elif isinstance(d, JavaObject):
        d = JNegmasGateway.gateway.entry_point.toMap(d)
        if deep:
            d = {python_identifier(k): from_java(v) for k, v in d.items()}
        else:
            d = {python_identifier(k): v for k, v in d.items()}
        deep = False
    if isinstance(d, dict):
        if remove_type_field:
            python_class_name = d.pop(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
        else:
            python_class_name = d.get(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
        if python_class_name is not None:
            python_class_name = py_class_name(python_class_name)
            if python_class_name.endswith("Issue"):
                python_class = get_class("negmas.outcomes.Issue")
            else:
                python_class = get_class(python_class_name)
            # we resolve sub-objects first from the dict if deep is specified before calling from_java on the class
            if deep:
                d = {
                    python_identifier(k): from_java(v)
                    for k, v in d.items()
                    if good_field(k)
                }
            # from_java needs to do a shallow conversion from a dict as deep conversion is taken care of already.
            if hasattr(python_class, "from_java"):
                return python_class.from_java(
                    {python_identifier(k): v for k, v in d.items()}, python_class_name
                )
            if deep:
                d = {
                    python_identifier(k): from_java(v)
                    for k, v in d.items()
                    if good_field(k)
                }
            else:
                d = {python_identifier(k): v for k, v in d.items() if good_field(k)}
            return python_class(**d)
        return d
    raise (ValueError(str(d)))


def to_dict(value, deep=True, add_type_field=True, camel=True):
    """Encodes the given value as nothing more complex than simple dict of either dicts, lists or builtin numeric
    or string values
    Args:
        value: Any object
        deep: Whether we should go deep in the encoding or do a shallow encoding
        add_type_field: Whether to add a type field. If True, A field named `PYTHON_CLASS_IDENTIFIER` will be added
        giving the type of `value`
        camel: Convert to camel_case if True
    Remarks:
        - All iterables are converted to lists when `deep` is true.
        - If the `value` object has a `to_java` member, it will be called to do the conversion, otherwise its `__dict__`
          or `__slots__` member will be used.
    See Also:
          `from_java`, `PYTHON_CLASS_IDENTIFIER`
    """

    def _j(s: str) -> str:
        if camel:
            return java_identifier(s)
        return str(s)

    def good_field(k: str):
        return (
            not k.startswith("python_")
            and not k.startswith("java")
            and not (k != PYTHON_CLASS_IDENTIFIER and k.startswith("_"))
        )

    if isinstance(value, JavaObject):
        return value
    if isinstance(value, dict):
        if not deep:
            return {_j(k): v for k, v in value.items()}
        return {
            _j(k): to_dict(v, add_type_field=add_type_field, camel=camel)
            for k, v in value.items()
            if good_field(k)
        }
    if isinstance(value, Iterable) and not deep:
        return value
    if isinstance(value, Iterable) and not isinstance(value, str):
        return [to_dict(_, add_type_field=add_type_field, camel=camel) for _ in value]
    if hasattr(value, "to_java"):
        converted = value.to_java()
        if isinstance(converted, dict):
            if add_type_field and (PYTHON_CLASS_IDENTIFIER not in converted.keys()):
                converted[PYTHON_CLASS_IDENTIFIER] = (
                    value.__class__.__module__ + "." + value.__class__.__name__
                )
            return {_j(k): v for k, v in converted.items()}
        else:
            return converted
    if hasattr(value, "__dict__"):
        if deep:
            d = {
                _j(k): to_dict(v, add_type_field=add_type_field, camel=camel)
                for k, v in value.__dict__.items()
                if good_field(k)
            }
        else:
            d = {_j(k): v for k, v in value.__dict__.items() if good_field(k)}
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )

        # ugly ugly ugly ugly
        if "_NamedObject__uuid" in value.__dict__:
            d["id"] = value.__dict__["_NamedObject__uuid"]
        if "_NamedObject__name" in value.__dict__:
            d["name"] = value.__dict__["_NamedObject__name"]

        return d
    if hasattr(value, "__slots__"):
        if deep:
            d = dict(
                zip(
                    (_j(k) for k in value.__slots__),
                    (
                        to_dict(
                            getattr(value, _),
                            add_type_field=add_type_field,
                            camel=camel,
                        )
                        for _ in value.__slots__
                    ),
                )
            )
        else:
            d = dict(
                zip(
                    (_j(k) for k in value.__slots__),
                    (getattr(value, _) for _ in value.__slots__),
                )
            )
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )
        return d
    # a builtin
    if isinstance(value, np.int64):
        return int(value)
    return value


def py_class_name(python_class_name: str) -> str:
    """
    Converts a class name that we got from Java to the corresponding class name in python

    Args:

        python_class_name: The class name we got from JNEgMAS

    Returns:

        The class name in negmas.

    """
    lst = python_class_name.split(".")
    if lst[-1].startswith("Python"):
        lst[-1] = lst[-1][len("Python") :]
    python_class_name = ".".join(lst)
    return python_class_name


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
        obj._java_object = java_object
        obj._connected = True
        obj._java_class_name = java_object.getClass().getSimpleName()
        return obj

    def init_java_bridge(
        self,
        java_object,
        java_class_name: str,
        auto_load_java: bool = False,
        python_shadow_object: Any = None,
    ):
        """
        initializes a connection to the java bridge creating a member called `java_object` that can be used to access
        the counterpart object in Java

        Args:
            java_object: A java object that already exists of the correct type. If given no new objects will be created
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
        self._java_class_name = (
            java_class_name
            if java_class_name is not None
            else java_object.getClass().getName()
        )
        self._connected = JNegmasGateway.connect(auto_load_java=auto_load_java)
        self._java_object = (
            java_object
            if java_object is not None
            else self._create(python_shadow_object)
        )

    def _create(self, python_shadow: Any = None):
        """
        Creates the internal object.

        Args:

              python_shadow: An object that has an internal Java class with implements = ['jnegmas.Callable']

        """
        if python_shadow is None:
            return JNegmasGateway.gateway.entry_point.create(self._java_class_name)
        return JNegmasGateway.gateway.entry_point.create(
            self._java_class_name, python_shadow
        )
