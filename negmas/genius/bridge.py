"""
Implements GeniusBridge which manages connections to Genius through Py4J.

The main class is the GeniusBridge class which encapsulates a gateway 
(connection) to a JVM running the geniusbridge.jar file.
We can have multiple gateways connected to the same JVM bridge.

The bridge life-cycle is as follows:
start -> connect -> stop/restart

The most important methods that this class provides are:

Status enquiry
--------------
- is_running: Tells you if a bridge is running on a given port

Bridge Lifetime Control
-----------------------
- start() starts a bridge and connects to it
- connect() connects to a *running* bridge
- stop() stops a running bridge. 
- restart() stops then starts a bridge.

Bridge Control Operations
-------------------------

- clean() removes all agents from the bridge and runs garbage collection. You 
  must be sure that no active negotiations are happening when you call this 
  method.
- kill_threads() [not recommended] Kills all threads started by the bridge. The 
  bridge is most likely going to become unusable after running this command


"""
import os
import pathlib
import socket
import subprocess
import time
import traceback
import warnings
from typing import Any, Dict, Optional

import psutil
from py4j.java_gateway import (
    CallbackServerParameters,
    GatewayParameters,
    JavaGateway,
    JavaObject,
)
from py4j.protocol import Py4JNetworkError

from ..config import CONFIG_KEY_GENIUS_BRIDGE_JAR, NEGMAS_CONFIG
from ..helpers import TimeoutCaller, TimeoutError, unique_name

from .common import DEFAULT_JAVA_PORT, get_free_tcp_port

__all__ = [
    "GeniusBridge",
    "init_genius_bridge",
    "genius_bridge_is_running",
]


def _kill_process(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def init_genius_bridge(
    path: str = None,
    port: int = DEFAULT_JAVA_PORT,
    debug: bool = False,
    timeout: float = 0,
) -> bool:
    """Initializes a genius connection

    Args:
        path: The path to a JAR file that runs negloader
        port: port number to use
        debug: If true, passes --debug to the bridge
        timeout: If positive and nonzero, passes it as the global timeout for the bridge. Note that
                 currently, the bridge supports only integer timeout values and the fraction will be
                 truncated.

    Returns:
        True if successful

    """
    if port <= 0:
        port = get_free_tcp_port()
    if genius_bridge_is_running(port):
        return True
    # if not force and common_gateway is not None and common_port == port:
    #     print("Java already initialized")
    #     return True

    if not path:
        path = NEGMAS_CONFIG.get(CONFIG_KEY_GENIUS_BRIDGE_JAR, None)
    if path is None:
        warnings.warn(
            "Cannot find the path to genius bridge jar. Download the jar somewhere in your machine and add its path"
            'to ~/negmas/config.json under the key "genius_bridge_jar".\n\nFor example, if you downloaded the jar'
            " to /path/to/your/jar then edit ~/negmas/config.json to read something like\n\n"
            '{\n\t"genius_bridge_jar": "/path/to/your/jar",\n\t.... rest of the config\n}\n\n'
            "You can find the jar at http://www.yasserm.com/scml/genius-8.0.4-bridge.jar"
        )
        return False
    path = pathlib.Path(path).expanduser().absolute()
    if debug:
        params = " --debug"
    else:
        params = ""
    if timeout >= 0:
        params += f" --timeout={int(timeout)}"

    try:
        subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
            f"java -jar {path} --die-on-exit {params} {port}", shell=True
        )
    except (OSError, TimeoutError, RuntimeError, ValueError):
        return False
    time.sleep(0.5)
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=port, auto_close=True),
        callback_server_parameters=CallbackServerParameters(
            port=0, daemonize=True, daemonize_connections=True
        ),
    )
    python_port = gateway.get_callback_server().get_listening_port()
    gateway.java_gateway_server.resetCallbackClient(
        gateway.java_gateway_server.getCallbackClient().getAddress(), python_port
    )
    return True


def genius_bridge_is_running(port: int = DEFAULT_JAVA_PORT) -> bool:
    """
    Checks whether a Genius Bridge is running. A genius bridge allows you to use `GeniusNegotiator` objects.

    Remarks:

        You can start a Genius Bridge in at least three ways:

        - execute the python function `init_genius_bridge()` in this module
        - run "negmas genius" on the terminal
        - execute `GeniusBridge.start()`

    """
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        # we know someone is listening. Now we check that it is us
        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, auto_close=True),
            callback_server_parameters=CallbackServerParameters(
                port=0, daemonize=True, daemonize_connections=True
            ),
        )
        gateway.jvm.System.currentTimeMillis()
        return True
    except ConnectionRefusedError:
        # try:
        #     s.shutdown(2)
        # except:
        #     pass
        s.close()
        return False
    except IndexError:
        # try:
        #     s.shutdown(2)
        # except:
        #     pass
        s.close()
        return False
    except Py4JNetworkError:
        # try:
        #     s.shutdown(2)
        # except:
        #     pass
        s.close()
        return False


class GeniusBridge:
    gateways: Dict[int, "GeniusBridge"] = dict()
    """Gateways to different genius bridges"""
    java_processes: Dict[int, Any] = dict()
    """handles to the java processes running the bridges"""
    python_ports: Dict[int, int] = dict()
    """The port used by python's Gateway to connect to the bridge"""

    def __init__(self,):
        raise RuntimeError(f"Cannot create objects of type GeniusBridge.")

    @classmethod
    def is_running(cls, port: int) -> bool:
        """Returns true if a geniusbridge.jar is running on the given port"""
        return genius_bridge_is_running(port)

    @classmethod
    def start(
        cls,
        port: int = DEFAULT_JAVA_PORT,
        path: str = None,
        debug: bool = False,
        timeout: float = 0,
        force_timeout: bool = True,
        save_logs: bool = True,
        log_path: Optional[os.PathLike] = None,
        die_on_exit: bool = False,
        use_shell: bool = False,
    ) -> int:
        """Initializes a genius connection

        Args:
            port: port number to use. A value <= 0 means get any free tcp port.
            path: The path to a JAR file that runs negloader
            debug: If true, passes --debug to the bridge
            timeout: If positive and nonzero, passes it as the global timeout for the bridge. Note that
                     currently, the bridge supports only integer timeout values and the fraction will be
                     truncated.
            force_timeout: if false, no timeout will be forced by the bridge
            save_logs: If false, the brige is instructed not to save any logs
            log_path: the path to store logs from the bridge. Onle effective if `save_logs`
                     If not given, defaults to ~/negmas/geniusbridge/logs/{port}-{datetime}.txt
            die_on_exit: If given, the bridge will be closed when this process is ended
            use_shell: If given, the bridge will be started in  a subshell.

        Returns:
           The port number used by the java process. 0 for failure 

        Remarks:
            - if a bridge is running, it will return its port and it does not matter whether 
              or not the bridge is started from this process or any other way.
            - it is recommended not to change the defaults for this function.

        """
        if port <= 0:
            port = get_free_tcp_port()
        if cls.is_running(port):
            return 0 if cls.gateway(port) is None else port
        path = (
            NEGMAS_CONFIG.get(CONFIG_KEY_GENIUS_BRIDGE_JAR, None)
            if path is None or not path
            else path
        )
        if path is None:
            warnings.warn(
                "Cannot find the path to genius bridge jar. Download the jar somewhere in your machine and add its path"
                'to ~/negmas/config.json under the key "genius_bridge_jar".\n\nFor example, if you downloaded the jar'
                " to /path/to/your/jar then edit ~/negmas/config.json to read something like\n\n"
                '{\n\t"genius_bridge_jar": "/path/to/your/jar",\n\t.... rest of the config\n}\n\n'
                "You can find the jar at http://www.yasserm.com/scml/geniusbridge.jar"
            )
            return 0
        path = pathlib.Path(path).expanduser().absolute()
        if log_path is None or not log_path:
            log_path = (
                pathlib.Path.home()
                / "negmas"
                / "geniusbridge"
                / "logs"
                / f"{unique_name(str(port), add_time=True, rand_digits=4, sep='.')}.txt"
            ).absolute()
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = pathlib.Path(log_path).absolute()
            log_path.parent.mkdir(parents=True, exist_ok=True)
        # if die_on_exit:
        #     if debug:
        #         params = " --die-on-exit --debug"
        #     else:
        #         params = " --die-on-exit"
        #     if force_timeout:
        #         params += " --force-timeout"
        #         if timeout >= 0:
        #             params += f" --timeout={int(timeout)}"
        #     else:
        #         params += " --no-timeout"
        #     params += " --with-logs" if save_logs else " --no-logs"
        #     if save_logs:
        #         params += f"--log-file={str(log_path)}"
        # else:
        if debug:
            params = ["--debug"]
        else:
            params = []
        if die_on_exit:
            params.append("--die-on-exit")
        if force_timeout:
            params.append("--force-timeout")
            if timeout >= 0:
                params.append(f"--timeout={int(timeout)}")
        else:
            params.append("--no-timeout")
        params.append("--with-logs" if save_logs else "--no-logs")
        if save_logs:
            params.append(f"--logfile={str(log_path)}")

        try:
            # if die_on_exit:
            #     cls.java_processes[port] = subprocess.Popen(
            #         f"java -jar {str(path)} {params} {port}",
            #         shell=use_shell,
            #         cwd=path.parent,
            #     )
            # else:
            cls.java_processes[port] = subprocess.Popen(
                ["java", "-jar", str(path)] + params + [f"{port}"],
                shell=use_shell,
                cwd=path.parent,
            )
        except (OSError, TimeoutError, RuntimeError, ValueError) as e:
            warnings.warn(str(e))
            return 0
        cls.wait_until_listening(port, timeout=0.1)
        return port if cls.gateway(port, force=True) is not None else 0

    @classmethod
    def gateway(cls, port=DEFAULT_JAVA_PORT, force=False):
        """
        Finds and returns a gateway for a genius bridge on the given port

        Args:
            port: The port used by the Jave genius bridge. 
            force: If true, a new gateway is created even if one exists in
                   the list of gateways available in `GeniusBridge`.gateways.
        Returns:
            The gateway if found otherwise an exception will be thrown

        Remarks:
            - this method does NOT start a bridge. It only connects to a
              running bridge.
        """
        assert port > 0
        gateway = cls.gateways.get(port, None) if not force else None
        if gateway is not None:
            return gateway
        try:
            gateway = JavaGateway(
                gateway_parameters=GatewayParameters(port=port, auto_close=True),
                callback_server_parameters=CallbackServerParameters(
                    port=0, daemonize=True, daemonize_connections=True
                ),
            )
            python_port = gateway.get_callback_server().get_listening_port()
            gateway.java_gateway_server.resetCallbackClient(
                gateway.java_gateway_server.getCallbackClient().getAddress(),
                python_port,
            )
        except:
            if gateway is not None:
                gateway.shutdown()
                gateway.shutdown_callback_server()
            return None
        cls.python_ports[port] = python_port
        cls.gateways[port] = gateway
        return gateway

    @classmethod
    def _close_gateway(cls, port):
        """
        Closes the gateway and removes it from all class dicts.
        """
        gateway = cls.gateways.get(port, None)
        if gateway is None:
            cls.java_processes.pop(port, None)
            cls.python_ports.pop(port, None)

        gateway.shutdown()
        gateway.shutdown_callback_server()
        cls.gateways.pop(port, None)
        cls.java_processes.pop(port, None)
        cls.python_ports.pop(port, None)

    @classmethod
    def close_gateway(cls, port=DEFAULT_JAVA_PORT):
        """
        Closes the gateway.

        Args:
            port: The port the gateway is connected to. 
                  If None, DEFAULT_JAVA_PORT is used.
        """
        if port is None:
            port = DEFAULT_JAVA_PORT
        cls._close_gateway(port)

    @classmethod
    def close_gateways(cls):
        """
        Closes all open gateways.
        """
        for p in cls.gateways.keys():
            cls._close_gateway(p)

    @classmethod
    def shutdown(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True,) -> bool:
        """
        Attempts to shutdown the bridge on that port.
        
        Args:
            port: The port to shutdown.  

        Remarks:
            - This is the cleanest way to close a java bridge and it simply sends a 
              message to the bridge to shut itself down and cleanly shuts down the 
              py4j bridge.
        """
        assert port > 0
        try:
            gateway = cls.gateway(port)
        except:
            return not genius_bridge_is_running(port)
        if gateway is None:
            return True
        try:
            gateway.entry_point.shutdown()
        except:
            pass
        if wait:
            cls.wait_until_not_listening(port)
        cls._close_gateway(port)
        return True

    @classmethod
    def stop(cls, port: int = DEFAULT_JAVA_PORT) -> bool:
        """Stops a running bridge

        Args:
            port: port number to use

        Returns:
            True if successful

        Remarks:
            - You should use this method to stop bridges.
            - It tries the following in order:
                1. shutdown the java bridge by calling its shutdown() method.
                2. killing the java bridge by calling its kill() method.
                3. killing the java bridge forcibly by killing the process
            - This method always waits for a short time to allow 
              each process to complete. If it returns True then 
              the bridge is no longer listening on the given port.

        """
        assert port > 0
        if not genius_bridge_is_running(port):
            return True
        cls.shutdown(port, wait=True)
        if not genius_bridge_is_running(port):
            return True
        cls.kill(port, wait=True)
        if not genius_bridge_is_running(port):
            return True
        cls.kill_forced(port, wait=True)
        if not genius_bridge_is_running(port):
            return True
        return False

    @classmethod
    def restart(cls, port: int = DEFAULT_JAVA_PORT, *args, **kwargs) -> bool:
        """Starts or restarts the genius bridge

        Args:
            port: port number to use

        Returns:
            True if successful

        """
        assert port > 0
        if not cls.stop(port):
            return False
        cls.wait_until_not_listening(port, 1)
        if not cls.start(port, *args, **kwargs):
            return False
        cls.wait_until_listening(port, 1)
        return genius_bridge_is_running(port)

    @classmethod
    def wait_until_listening(
        cls, port: int = DEFAULT_JAVA_PORT, timeout: float = 0.5,
    ) -> bool:
        """
        waits until a genius bridge is  listening to the given port

        Args:
            port: The port to test
            timeout: Maximum time to wait before returning (in seconds)

        Returns:
            True if the genius bridge is running any more (success).
        """
        if genius_bridge_is_running(port):
            return True
        time.sleep(timeout)
        return TimeoutCaller.run(
            lambda: genius_bridge_is_running(port), timeout=timeout
        )

    @classmethod
    def wait_until_not_listening(
        cls, port: int = DEFAULT_JAVA_PORT, timeout: float = 0.5,
    ) -> bool:
        """
        waits until the genius bridge is not listening to the given port

        Args:
            port: The port to test
            max_sleep: Maximum time to wait before returning (in seconds)

        Returns:
            True if the genius bridge is NOT running any more (success).
        """
        if not genius_bridge_is_running(port):
            return False
        time.sleep(timeout)
        return not genius_bridge_is_running(port)

    @classmethod
    def kill_threads(
        cls, port: int = DEFAULT_JAVA_PORT, wait_time: float = 0.5,
    ) -> bool:
        """kills all threads in the given java bridge"""
        assert port > 0
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            warnings.warn(
                f"Failed to kill threads at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            return False
        if gateway is None:
            return False
        try:
            gateway.entry_point.kill_threads(int(wait_time * 1000))
        except:
            pass
        return True

    @classmethod
    def kill(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True,) -> bool:
        """Kills the java bridge connected to this port by asking it to exit"""
        assert port > 0
        try:
            gateway = cls.gateway(port, force=True)
        except:
            return False
        if gateway is None:
            return False
        try:
            gateway.entry_point.kill()
        except:
            pass
        if wait:
            cls.wait_until_not_listening(port)
        cls._close_gateway(port)
        return True

    @classmethod
    def kill_forced(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True,) -> bool:
        """Kills the java bridge connected to this port forcibly.

        Remarks:
            - The java bridge process must have been started by this process.
        """
        assert port > 0
        p = cls.java_processes.pop(port, None)
        if p is None:
            warnings.warn(
                f"Attempting to force-kill a genius brdige we did not start "
                "at port {port}"
            )
            return False
        _kill_process(p.pid)
        if wait:
            cls.wait_until_not_listening(port)
        cls._close_gateway(port)
        return True

    @classmethod
    def clean(cls, port=DEFAULT_JAVA_PORT) -> bool:
        """
        Removes all agents and runs garbage collection on the bridge
        """
        assert port > 0
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            warnings.warn(
                f"Failed to kill threads at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            return False
        if gateway is None:
            return False
        gateway.entry_point.clean()
        return True

    @classmethod
    def clean_all(cls) -> bool:
        """
        Removes all agents and runs garbage collection on all bridges.
        """
        success = True
        for port in cls.gateways.keys():
            success = success and cls.clean(port)
        return success

    @classmethod
    def connect(cls, port: int = DEFAULT_JAVA_PORT) -> JavaObject:
        """
        Connects to a running genius-bridge

        Args:
            port: The port at which the bridge in listening in Java

        Remarks:
            - The difference between this method and start() is that this one 
              does not attempt to start a java bridge if one does not exist.

        """
        assert port > 0
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            warnings.warn(
                f"Failed to kill threads at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise RuntimeError(e)
        if gateway is None:
            return None
        return gateway.entry_point
