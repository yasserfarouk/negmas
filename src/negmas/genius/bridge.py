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
- is_installed: Tells you if the bridge jar is insalled in the default location

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

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import negmas.warnings as warnings

from ..config import CONFIG_KEY_GENIUS_BRIDGE_JAR, negmas_config
from ..helpers import TimeoutCaller, TimeoutError, unique_name
from .common import DEFAULT_JAVA_PORT, get_free_tcp_port

if TYPE_CHECKING:
    from py4j.java_gateway import JavaGateway, JavaObject

__all__ = [
    "GeniusBridge",
    "init_genius_bridge",
    "genius_bridge_is_running",
    "genius_bridge_is_installed",
    "run_native_genius_negotiation",
]

GENIUS_LOG_BASE = Path(
    negmas_config("genius_log_base", Path.home() / "negmas" / "geniusbridge" / "logs")  # type: ignore
)


def _kill_process(proc_pid):
    import psutil

    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def genius_bridge_is_running(port: int = DEFAULT_JAVA_PORT) -> bool:
    """
    Checks whether a Genius Bridge is running. A genius bridge allows you to use `GeniusNegotiator` objects.

    Remarks:

        You can start a Genius Bridge in at least three ways:

        - execute the python function `init_genius_bridge()` in this module
        - run "negmas genius" on the terminal
        - execute `GeniusBridge.start()`

    """
    from py4j.java_gateway import (
        CallbackServerParameters,
        GatewayParameters,
        JavaGateway,
    )
    from py4j.protocol import Py4JNetworkError

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
        gateway.jvm.System.currentTimeMillis()  # type: ignore
        return True
    except ConnectionRefusedError:
        # try:
        #     s.shutdown(2)
        # except Exception:
        #     pass
        s.close()
        return False
    except IndexError:
        # try:
        #     s.shutdown(2)
        # except Exception:
        #     pass
        s.close()
        return False
    except Py4JNetworkError:
        # try:
        #     s.shutdown(2)
        # except Exception:
        #     pass
        s.close()
        return False


def init_genius_bridge(
    path: Path | str | None = None,
    port: int = DEFAULT_JAVA_PORT,
    debug: bool = False,
    timeout: float = 0,
    force_timeout: bool = True,
    save_logs: bool = False,
    log_path: os.PathLike | None = None,
    die_on_exit: bool = False,
    use_shell: bool = False,
    verbose: bool = False,
    allow_agent_print: bool = False,
    # capture_output: bool = False,
) -> int:
    """Initializes a genius connection

    Args:
        path: The path to a JAR file that runs negloader
        port: port number to use
        debug: If true, passes --debug to the bridge
        timeout: If positive and nonzero, passes it as the global timeout for the bridge. Note that
                 currently, the bridge supports only integer timeout values and the fraction will be
                 truncated.

    Returns:
        Port if started. -1 if a bridge is already running on this port. 0 if failed

    """
    if port <= 0:
        port = get_free_tcp_port()
    if genius_bridge_is_running(port):
        return -1
    port = GeniusBridge.start(
        port=port,
        debug=debug,
        timeout=timeout,
        path=str(path) if path else path,
        force_timeout=force_timeout,
        save_logs=save_logs,
        log_path=log_path,
        die_on_exit=die_on_exit,
        use_shell=use_shell,
        verbose=verbose,
        allow_agent_print=allow_agent_print,
        # capture_output=capture_output,
    )
    return port
    # # if not force and common_gateway is not None and common_port == port:
    # #     print("Java already initialized")
    # #     return True
    #
    # if not path:
    #     path = negmas_config(  # type: ignore
    #         CONFIG_KEY_GENIUS_BRIDGE_JAR,
    #         pathlib.Path.home() / "negmas" / "files" / "geniusbridge.jar",
    #     )
    # if path is None:
    #     warnings.warn(
    #         "Cannot find the path to genius bridge jar. Download the jar somewhere in your machine and add its path"
    #         'to ~/negmas/config.json under the key "genius_bridge_jar".\n\nFor example, if you downloaded the jar'
    #         " to /path/to/your/jar then edit ~/negmas/config.json to read something like\n\n"
    #         '{\n\t"genius_bridge_jar": "/path/to/your/jar",\n\t.... rest of the config\n}\n\n'
    #         "You can find the jar at https://yasserfarouk.github.io/files/geniusbridge.jar",
    #         warnings.NegmasBridgePathWarning,
    #     )
    #     return False
    # path = Path(path).expanduser().absolute()
    # if debug:
    #     params = " --debug"
    # else:
    #     params = " --silent --no-logs"
    # if timeout >= 0:
    #     params += f" --timeout={int(timeout)}"
    #
    # try:
    #     subprocess.Popen(  # ['java', '-jar',  path, '--die-on-exit', f'{port}']
    #         f"java -jar {path} --die-on-exit {params} {port}", shell=True
    #     )
    # except (OSError, TimeoutError, RuntimeError, ValueError):
    #     return False
    # time.sleep(0.5)
    # gateway = JavaGateway(
    #     gateway_parameters=GatewayParameters(port=port, auto_close=True),
    #     callback_server_parameters=CallbackServerParameters(
    #         port=0, daemonize=True, daemonize_connections=True
    #     ),
    # )
    # python_port = gateway.get_callback_server().get_listening_port()  # type: ignore
    # gateway.java_gateway_server.resetCallbackClient(  # type: ignore
    #     gateway.java_gateway_server.getCallbackClient().getAddress(),  # type: ignore
    #     python_port,  # type: ignore
    # )
    # return True


def genius_bridge_is_installed() -> bool:
    """
    Checks if geniusbridge is available in the default path location
    """
    return (Path.home() / "negmas" / "files" / "geniusbridge.jar").exists()


class GeniusBridge:
    """GeniusBridge implementation."""

    gateways: dict[int, Any] = dict()
    java_processes: dict[int, Any] = dict()
    python_ports: dict[int, int] = dict()

    def __init__(self):
        """Initialize the instance."""
        raise RuntimeError("Cannot create objects of type GeniusBridge.")

    """Gateways to different genius bridges"""
    """handles to the java processes running the bridges"""
    """The port used by python's Gateway to connect to the bridge"""

    @classmethod
    def is_running(cls, port: int) -> bool:
        """Returns true if a geniusbridge.jar is running on the given port"""
        return genius_bridge_is_running(port)

    @classmethod
    def is_installed(cls) -> bool:
        """Returns true if a geniusbridge.jar is available"""
        return genius_bridge_is_installed()

    @classmethod
    def gateway(cls, port=DEFAULT_JAVA_PORT, force=False) -> JavaGateway | None:
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
        from py4j.java_gateway import (
            CallbackServerParameters,
            GatewayParameters,
            JavaGateway,
        )

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
            python_port = gateway.get_callback_server().get_listening_port()  # type: ignore
            gateway.java_gateway_server.resetCallbackClient(  # type: ignore
                gateway.java_gateway_server.getCallbackClient().getAddress(),  # type: ignore
                python_port,
            )
        except Exception:
            if gateway is not None:
                gateway.shutdown()
                gateway.shutdown_callback_server()
            return None
        cls.python_ports[port] = python_port
        cls.gateways[port] = gateway
        return gateway

    @classmethod
    def wait_until_listening(
        cls, port: int = DEFAULT_JAVA_PORT, timeout: float = 0.5
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
    def start(
        cls,
        port: int = DEFAULT_JAVA_PORT,
        path: str | None = None,
        debug: bool = False,
        timeout: float = 0,
        force_timeout: bool = True,
        save_logs: bool = False,
        log_path: os.PathLike | None = None,
        die_on_exit: bool = False,
        use_shell: bool = False,
        verbose: bool = False,
        allow_agent_print: bool = False,
        # capture_output: bool = False,
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
        path = (  # type: ignore
            negmas_config(
                CONFIG_KEY_GENIUS_BRIDGE_JAR,
                pathlib.Path.home() / "negmas" / "files" / "geniusbridge.jar",
            )
            if path is None or not path
            else path
        )
        if path is None:
            warnings.warn(
                "Cannot find the path to genius bridge jar. Download the jar somewhere in your machine and add its path"
                'to ~/negmas/config.json under the key "genius_bridge_jar".\n\nFor example, if you downloaded the jar'
                " to /path/to/your/jar then edit ~/negmas/config.json to read something like\n\n"
                '{\n\t"genius_bridge_jar": "/path/to/your/jar",\n\t.... rest of the config\n}\n\n'
                "You can find the jar at https://yasserfarouk.github.io/files/geniusbridge.jar",
                warnings.NegmasBridgePathWarning,
            )
            return 0
        path = Path(path).expanduser().absolute()  # type: ignore
        if log_path is None or not log_path:
            log_path = (
                GENIUS_LOG_BASE
                / f"{unique_name(str(port), add_time=True, rand_digits=4, sep='.')}.txt"
            ).absolute()
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(log_path).absolute()
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
        params.append("--verbose" if verbose else "--silent")
        if allow_agent_print:
            params.append("--allow-agent-print")

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
                # capture_output=capture_output,
                cwd=path.parent,  # type: ignore
            )
        except (OSError, TimeoutError, RuntimeError, ValueError) as e:
            warnings.warn(str(e), warnings.NegmasBridgeProcessWarning)
            return 0
        cls.wait_until_listening(port, timeout=0.1)
        return port if cls.gateway(port, force=True) is not None else 0

    @classmethod
    def _close_gateway(cls, port):
        """
        Closes the gateway and removes it from all class dicts.
        """
        gateway = cls.gateways.get(port, None)
        if gateway is None:
            cls.java_processes.pop(port, None)
            cls.python_ports.pop(port, None)

        gateway.shutdown()  # type: ignore
        gateway.shutdown_callback_server()  # type: ignore
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
        keys = list(cls.gateways.keys())
        for p in keys:
            cls._close_gateway(p)

    @classmethod
    def wait_until_not_listening(
        cls, port: int = DEFAULT_JAVA_PORT, timeout: float = 0.5
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
    def shutdown(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True) -> bool:
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
        except Exception:
            return not genius_bridge_is_running(port)
        if gateway is None:
            return True
        try:
            gateway.entry_point.shutdown()  # type: ignore
        except Exception:
            pass
        if wait:
            cls.wait_until_not_listening(port)
        cls._close_gateway(port)
        return True

    @classmethod
    def kill(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True) -> bool:
        """Kills the java bridge connected to this port by asking it to exit"""
        assert port > 0
        try:
            gateway = cls.gateway(port, force=True)
        except Exception:
            return False
        if gateway is None:
            return False
        try:
            gateway.entry_point.kill()  # type: ignore
        except Exception:
            pass
        if wait:
            cls.wait_until_not_listening(port)
        cls._close_gateway(port)
        return True

    @classmethod
    def kill_forced(cls, port: int = DEFAULT_JAVA_PORT, wait: bool = True) -> bool:
        """Kills the java bridge connected to this port forcibly.

        Remarks:
            - The java bridge process must have been started by this process.
        """
        assert port > 0
        p = cls.java_processes.pop(port, None)
        if p is None:
            warnings.warn(
                "Attempting to force-kill a genius bridge we did not start "
                "at port {port}",
                warnings.NegmasBridgeProcessWarning,
            )
            return False
        _kill_process(p.pid)
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
    def kill_threads(
        cls, port: int = DEFAULT_JAVA_PORT, wait_time: float = 0.5
    ) -> bool:
        """kills all threads in the given java bridge"""
        assert port > 0
        try:
            gateway = cls.gateway(port, force=True)
        except Exception as e:
            warnings.warn(
                f"Failed to kill threads at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}",
                warnings.NegmasBridgeProcessWarning,
            )
            return False
        if gateway is None:
            return False
        try:
            gateway.entry_point.kill_threads(int(wait_time * 1000))  # type: ignore
        except Exception:
            pass
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
                f"{traceback.format_exc()}",
                warnings.NegmasBridgeProcessWarning,
            )
            return False
        if gateway is None:
            return False
        gateway.entry_point.clean()  # type: ignore
        return True

    @classmethod
    def get_active_agents(cls, port: int = DEFAULT_JAVA_PORT) -> list[str]:
        """
        Gets all active agent UUIDs from the bridge.

        Args:
            port: The port at which the bridge is listening in Java

        Returns:
            A list of agent UUIDs currently active in the bridge

        Remarks:
            - Thread-safe via ConcurrentHashMap on the Java side
            - Returns empty list if bridge is not running or connection fails
        """
        assert port > 0
        try:
            gateway = cls.gateway(port, force=False)
        except Exception as e:
            warnings.warn(
                f"Failed to get active agents at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}",
                warnings.NegmasBridgeProcessWarning,
            )
            return []
        if gateway is None:
            return []
        try:
            result = gateway.entry_point.getActiveAgents()  # type: ignore
            if result is None:
                return []
            # The Java side returns agents separated by ENTRY_SEP
            # Split and filter out empty strings
            return [uuid.strip() for uuid in str(result).split("\n") if uuid.strip()]
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve active agents: {str(e)}",
                warnings.NegmasBridgeProcessWarning,
            )
            return []

    @classmethod
    def agent_exists(cls, uuid: str, port: int = DEFAULT_JAVA_PORT) -> bool:
        """
        Checks if a specific agent exists in the bridge.

        Args:
            uuid: The UUID of the agent to check
            port: The port at which the bridge is listening in Java

        Returns:
            True if the agent exists, False otherwise

        Remarks:
            - Thread-safe via ConcurrentHashMap on the Java side
        """
        assert port > 0
        if not uuid:
            return False
        try:
            gateway = cls.gateway(port, force=False)
        except Exception as e:
            warnings.warn(
                f"Failed to check agent existence at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}",
                warnings.NegmasBridgeProcessWarning,
            )
            return False
        if gateway is None:
            return False
        try:
            return bool(gateway.entry_point.agentExists(uuid))  # type: ignore
        except Exception as e:
            warnings.warn(
                f"Failed to check if agent {uuid} exists: {str(e)}",
                warnings.NegmasBridgeProcessWarning,
            )
            return False

    @classmethod
    def get_agent_count(cls, port: int = DEFAULT_JAVA_PORT) -> int:
        """
        Gets the number of active agents in the bridge.

        Args:
            port: The port at which the bridge is listening in Java

        Returns:
            The number of active agents, or 0 if bridge is not running

        Remarks:
            - Thread-safe via ConcurrentHashMap on the Java side
        """
        assert port > 0
        try:
            gateway = cls.gateway(port, force=False)
        except Exception as e:
            warnings.warn(
                f"Failed to get agent count at port {port} with error: {str(e)}\n"
                f"{traceback.format_exc()}",
                warnings.NegmasBridgeProcessWarning,
            )
            return 0
        if gateway is None:
            return 0
        try:
            return int(gateway.entry_point.getAgentCount())  # type: ignore
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve agent count: {str(e)}",
                warnings.NegmasBridgeProcessWarning,
            )
            return 0

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
                f"{traceback.format_exc()}",
                warnings.NegmasBridgeProcessWarning,
            )
            raise RuntimeError(e)
        if gateway is None:
            raise RuntimeError("Got None as the gateway!!")
        return gateway.entry_point


def run_native_genius_negotiation(
    negotiators: list[str | type],
    scenario: Scenario,
    n_steps: int = -1,
    time_limit: int = -1,
    mechanism_type: str = "genius.core.protocol.StackedAlternatingOffersProtocol",
    port: int = DEFAULT_JAVA_PORT,
    auto_start_bridge: bool = True,
    trace_mode: str = "none",
) -> SAOState:
    """
    Runs a negotiation natively inside Genius using the genius bridge.

    Args:
        negotiators: List of negotiator specifications. Each can be:
            - A string with Java class name (e.g., "agents.anac.y2015.Atlas3.Atlas3")
            - A GeniusNegotiator type
        scenario: The negotiation scenario containing issues and ufuns
        n_steps: Number of negotiation rounds (must specify either n_steps or time_limit, not both)
        time_limit: Time limit in seconds (must specify either n_steps or time_limit, not both)
        mechanism_type: Java class name of the negotiation protocol/mechanism
        port: Port for the genius bridge
        auto_start_bridge: If True, automatically start the genius bridge if not running

    Returns:
        SAOState: Final state of the negotiation with agreement and trace information

    Raises:
        ValueError: If both n_steps and time_limit are specified or neither is specified
        ValueError: If number of negotiators doesn't match number of ufuns in scenario
        RuntimeError: If the genius bridge is not running and auto_start_bridge is False

    Examples:
        >>> from negmas import load_genius_domain_from_folder
        >>> scenario = load_genius_domain_from_folder("path/to/domain")
        >>> negotiators = [
        ...     "agents.anac.y2015.Atlas3.Atlas3",
        ...     "agents.anac.y2015.RandomDance.RandomDance",
        ... ]
        >>> state = run_native_genius_negotiation(negotiators, scenario, n_steps=100)
        >>> print(f"Agreement: {state.agreement}")
        >>> print(f"Steps: {state.step}")
    """
    from ..sao.common import SAOState
    from .negotiator import GeniusNegotiator
    import tempfile
    from pathlib import Path

    # Validate inputs
    if (n_steps >= 0 and time_limit >= 0) or (n_steps < 0 and time_limit < 0):
        raise ValueError(
            "Must specify exactly one of n_steps or time_limit (not both, not neither)"
        )

    if len(negotiators) != scenario.n_negotiators:
        raise ValueError(
            f"Number of negotiators ({len(negotiators)}) must match "
            f"number of ufuns in scenario ({scenario.n_negotiators})"
        )

    if trace_mode not in ("none", "full", "full_with_utils"):
        raise ValueError(
            f"trace_mode must be 'none', 'full', or 'full_with_utils', got '{trace_mode}'"
        )

    # Ensure bridge is running
    if not genius_bridge_is_running(port):
        if not auto_start_bridge:
            raise RuntimeError(
                f"Genius bridge is not running on port {port}. "
                "Set auto_start_bridge=True to start it automatically."
            )
        init_genius_bridge(port=port)

    # Get gateway
    gateway = GeniusBridge.gateway(port, force=True)
    if gateway is None:
        raise RuntimeError(f"Failed to connect to genius bridge on port {port}")

    # Convert negotiators to Java class names
    agent_class_names = []
    for neg in negotiators:
        if isinstance(neg, str):
            agent_class_names.append(neg)
        elif isinstance(neg, type) and issubclass(neg, GeniusNegotiator):
            # Extract java_class_name from GeniusNegotiator class
            if hasattr(neg, "java_class_name"):
                agent_class_names.append(neg.java_class_name)
            else:
                raise ValueError(
                    f"GeniusNegotiator type {neg} does not have java_class_name attribute"
                )
        else:
            raise ValueError(
                f"Negotiator must be a string (Java class name) or GeniusNegotiator type, got {type(neg)}"
            )

    # Create temporary files for domain and profiles
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        domain_file = tmpdir_path / "domain.xml"
        profile_files = [
            tmpdir_path / f"profile{i}.xml" for i in range(len(negotiators))
        ]

        # Save scenario to Genius XML files
        scenario.to_genius_files(domain_file, profile_files)

        # Prepare parameters for run_negotiation
        agents_str = ";".join(agent_class_names)
        profiles_str = ";".join(str(f) for f in profile_files)
        output_file = str(tmpdir_path / "negotiation_log.txt")

        # Run the negotiation in Genius
        try:
            if trace_mode == "none":
                success = gateway.entry_point.run_negotiation(  # type: ignore
                    mechanism_type,
                    str(domain_file),
                    profiles_str,
                    agents_str,
                    output_file,
                    n_steps if n_steps >= 0 else -1,
                    time_limit if time_limit >= 0 else -1,
                )
                trace_str = ""
            else:
                success = gateway.entry_point.run_negotiation_with_trace(  # type: ignore
                    mechanism_type,
                    str(domain_file),
                    profiles_str,
                    agents_str,
                    output_file,
                    n_steps if n_steps >= 0 else -1,
                    time_limit if time_limit >= 0 else -1,
                    trace_mode,
                )
                # Get the trace data
                trace_str = gateway.entry_point.get_last_trace()  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to run negotiation in Genius: {e}")

        if not success:
            raise RuntimeError("Negotiation failed in Genius (returned False)")

        # Parse trace if collected
        from ..common import TraceElement

        ENTRY_SEP = "<<y,y>>"
        FIELD_SEP = "<<sy>>"

        trace_history = []
        agreement = None
        final_step = n_steps if n_steps >= 0 else -1
        final_state = "ended"

        if trace_str:
            # Parse trace string
            entries = trace_str.split(ENTRY_SEP) if trace_str else []

            for entry in entries:
                if not entry:
                    continue

                fields = entry.split(FIELD_SEP)
                if len(fields) < 7:
                    continue  # Malformed entry

                try:
                    time = float(fields[0])
                    relative_time = float(fields[1])
                    step = int(fields[2])
                    negotiator = fields[3]
                    offer_str = fields[4]
                    fields[5]
                    state = fields[6]

                    # Track final step and state
                    if step > final_step:
                        final_step = step
                    final_state = state

                    # Parse offer to outcome if present
                    offer = None
                    if offer_str and offer_str.strip():
                        # TODO: Parse offer string to actual outcome
                        # For now, store as string
                        pass

                    # Determine responses - for Genius, this is simplified
                    responses = {}

                    # Create trace element
                    trace_element = TraceElement(
                        time=time,
                        relative_time=relative_time,
                        step=step,
                        negotiator=negotiator,
                        offer=offer,
                        responses=responses,
                        state=state,
                        text=None,
                        data=None,
                    )
                    trace_history.append(trace_element)

                    # If state is agreement, last offer is the agreement
                    if state == "agreement" and offer_str:
                        # TODO: Parse to actual outcome
                        pass

                except (ValueError, IndexError):
                    # Skip malformed entries
                    continue

        # Try to read the log file to find agreement information (fallback)
        if Path(output_file).exists() and agreement is None:
            try:
                with open(output_file, "r") as f:
                    f.read()
                    # Parse log to find agreement
                    # This is a simplified parser - the actual format depends on Genius output
                    pass
            except Exception:
                pass

        # Determine if timed out or ended normally
        timedout = final_state == "timedout"

        # Create and return SAOState
        state = SAOState(
            running=False,
            waiting=False,
            started=True,
            step=final_step,
            time=0.0,
            relative_time=1.0,
            broken=not success,
            timedout=timedout,
            agreement=agreement,
            results=None,
            n_negotiators=len(negotiators),
            has_error=not success,
            error_details="" if success else "Negotiation failed in Genius",
            threads={},
            last_thread="",
            current_offer=None,
            current_proposer=None,
            current_proposer_agent=None,
            n_acceptances=0,
            new_offers=[],
            new_offerer_agents=[],
            last_negotiator=None,
            current_data=None,
            new_data=[],
        )

        # Add trace history if available
        if trace_history:
            # Note: SAOState doesn't have a history attribute by default
            # We'll attach it as a custom attribute for compatibility
            state._history = trace_history  # type: ignore

        return state
