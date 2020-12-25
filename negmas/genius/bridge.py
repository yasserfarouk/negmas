"""
Implements GeniusBridge which manages connections to Genius through Py4J.
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

DEFAULT_JAVA_PORT = 25337
DEFAULT_PYTHON_PORT = 25338  # not currently being used

__all__ = ["GeniusBridge", "init_genius_bridge", "genius_bridge_is_running"]


def _kill_process(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def init_genius_bridge(
    path: str = None, port: int = 0, debug: bool = False, timeout: float = 0,
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
    port = port if port > 0 else DEFAULT_JAVA_PORT
    if genius_bridge_is_running(port):
        return True
    # if not force and common_gateway is not None and common_port == port:
    #     print("Java already initialized")
    #     return True

    if not path:
        path = NEGMAS_CONFIG.get(CONFIG_KEY_GENIUS_BRIDGE_JAR, None)
    if path is None:
        print(
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


def genius_bridge_is_running(port: int = None) -> bool:
    """
    Checks whether a Genius Bridge is running. A genius bridge allows you to use `GeniusNegotiator` objects.

    Remarks:

        You can start a Genius Bridge in at least two ways:

        - execute the python function `init_genius_bridge()` in this module
        - run "negmas genius" on the terminal

    """
    if port is None:
        port = DEFAULT_JAVA_PORT
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.connect(("127.0.0.1", port))
        try:
            s.shutdown(2)
        except:
            pass
        s.close()
        return True
    except ConnectionRefusedError:
        try:
            s.shutdown(2)
        except:
            pass
        s.close()
        return False
    except IndexError:
        try:
            s.shutdown(2)
        except:
            pass
        s.close()
        return False
    except Py4JNetworkError:
        try:
            s.shutdown(2)
        except:
            pass
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
    def is_running(cls, port):
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
    ) -> bool:
        """Initializes a genius connection

        Args:
            path: The path to a JAR file that runs negloader
            port: port number to use
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
            True if successful

        """
        port = port if port is not None and port > 0 else DEFAULT_JAVA_PORT
        if genius_bridge_is_running(port):
            return cls.gateway(port)
        # if not force and common_gateway is not None and common_port == port:
        #     print("Java already initialized")
        #     return True

        path = (
            NEGMAS_CONFIG.get(CONFIG_KEY_GENIUS_BRIDGE_JAR, None)
            if path is None or not path
            else path
        )
        if path is None:
            print(
                "Cannot find the path to genius bridge jar. Download the jar somewhere in your machine and add its path"
                'to ~/negmas/config.json under the key "genius_bridge_jar".\n\nFor example, if you downloaded the jar'
                " to /path/to/your/jar then edit ~/negmas/config.json to read something like\n\n"
                '{\n\t"genius_bridge_jar": "/path/to/your/jar",\n\t.... rest of the config\n}\n\n'
                "You can find the jar at http://www.yasserm.com/scml/geniusbridge.jar"
            )
            return False
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
            print(str(e))
            return False
        cls.wait_until_listening(port, timeout=0.1)
        return cls.gateway(port, force=True) is not None

    @classmethod
    def gateway(cls, port=DEFAULT_JAVA_PORT, force=False):
        """
        Finds and returns a gateway for a genius bridge on the given port

        Args:
            port: The port used by the Jave genius bridge
            force: If true, a new gateway is created even if one exists in
                   the list of gateways available in `GeniusBridge`.gateways.
        Returns:
            The gateway if found otherwise an exception will be thrown

        Remarks:
            - this method does NOT start a bridge. It only connects to a
              running bridge.
        """
        if port is None:
            port = DEFAULT_JAVA_PORT
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
            return None
        cls.python_ports[port] = python_port
        cls.gateways[port] = gateway
        return gateway

    @classmethod
    def _close_gateway(cls, port):
        gateway = cls.gateways.get(port, None)
        if gateway is None:
            cls.java_processes.pop(port, None)
            cls.python_ports.pop(port, None)

        try:
            gateway.shutdown(raise_exception=False)
        except:
            pass
        cls.gateways.pop(port, None)
        cls.java_processes.pop(port, None)
        cls.python_ports.pop(port, None)

    @classmethod
    def close_gateway(cls, port=DEFAULT_JAVA_PORT):
        if port is None:
            port = DEFAULT_JAVA_PORT
        cls._close_gateway(port)

    @classmethod
    def close_gateways(cls):
        for p in cls.gateways.keys():
            cls._close_gateway(p)

    @classmethod
    def shutdown(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True,) -> bool:
        """Attempts to shutdown the bridge on that port"""
        if port is None:
            port = DEFAULT_JAVA_PORT
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
    def restart(cls, port: int = DEFAULT_JAVA_PORT, *args, **kwargs) -> bool:
        """Starts or restarts the genius bridge

        Args:
            port: port number to use
            kwargs: Same arguments as `start`.

        Returns:
            True if successful

        """

        def _start_and_wait(port, *args, **kwargs):
            cls.start(port, *args, **kwargs)
            cls.wait_until_listening(port, 1)
            return genius_bridge_is_running(port)

        if not genius_bridge_is_running(port):
            return _start_and_wait(port, *args, **kwargs)
        cls.shutdown(port, wait=True)
        if not genius_bridge_is_running(port):
            return _start_and_wait(port, *args, **kwargs) is not None
        cls.kill(port, wait=True)
        if not genius_bridge_is_running(port):
            return _start_and_wait(port, *args, **kwargs) is not None
        cls.kill_forced(port, wait=True)
        if not genius_bridge_is_running(port):
            return _start_and_wait(port, *args, **kwargs) is not None
        raise RuntimeError("Cannot close the currently running bridge")

    @classmethod
    def wait_until_listening(
        cls, port: int = DEFAULT_JAVA_PORT, timeout: float = 0.5,
    ) -> bool:
        """
        waits until the genius bridge is  listening to the given port

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
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            print(
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
        """Kills the java bridge connected to this port"""
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
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            print(
                f"Failed to kill threads at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            return False
        if gateway is None:
            return False
        gateway.entry_point.clean()
        return True

    @classmethod
    def connect(cls, port: int = DEFAULT_JAVA_PORT) -> JavaObject:
        """
        Connects to a running genius-bridge

        Args:
            port: The port at which the bridge in listening in Java

        """
        if port is None:
            port = DEFAULT_JAVA_PORT
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            print(
                f"Failed to kill threads at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise RuntimeError(e)
        if gateway is None:
            return None
        return gateway.entry_point
