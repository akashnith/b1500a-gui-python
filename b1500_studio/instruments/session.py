"""Hang-proof WGFMU session: the DLL is driven from a worker subprocess.

Why this exists
---------------
``WGFMU_openSession()`` opens its own VISA session inside ``wgfmu.dll``.  When
the Keysight IO Libraries services are not running, that call blocks inside
native code **forever**: there is no timeout parameter and no way to abort it.
A thread cannot be killed in Python, so an in-process call freezes the GUI
permanently -- which is exactly what the Qt app does.

Running the DLL in a child process makes the hang recoverable: the parent
waits with a timeout and, on expiry, calls ``terminate()``, which on Windows
is ``TerminateProcess`` and kills the process even while it sits inside a DLL.

:class:`RemoteWgfmu` implements the same :class:`~b1500_studio.driver.Driver`
interface as the in-process :class:`~b1500_studio.wgfmu_dll.Wgfmu`, so the GUI
and CLI can use either.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from typing import List, Optional

from ..core.driver import Driver, DriverError, Hardware, LogFn, Results
from ..core.plan import Plan
from .wgfmu_dll import DEFAULT_ADDRESS, DEFAULT_DLL, WgfmuError

__all__ = ["RemoteWgfmu", "WgfmuHang", "VISA_HANG_HELP"]

VISA_HANG_HELP = """\
wgfmu.dll stopped responding inside the Keysight VISA layer.

This is a Windows VISA problem, not a problem with this program or with the
instrument.  The usual cause is that the Keysight IO Libraries services are
not running, so VISA never answers and the DLL waits forever.

Fix, in an Administrator PowerShell:

  Set-Service AgilentIOLibrariesService -StartupType Automatic
  Set-Service KeysightDistributedInfrastructureService -StartupType Automatic
  Start-Service KeysightDistributedInfrastructureService
  Start-Service AgilentIOLibrariesService

Then, if NI-VISA was installed after the Keysight IO Libraries Suite, make
Keysight the primary VISA again and reboot:

  & "C:\\Program Files\\Keysight\\IO Libraries Suite\\primary-visa-utility.exe"

Cross-check with Keysight Connection Expert: if it cannot list the GPIB
interface either, Python is not the problem.
"""

#: Default timeouts in seconds.
OPEN_TIMEOUT = 30.0
CALL_TIMEOUT = 30.0
CLOSE_TIMEOUT = 10.0


class WgfmuHang(DriverError):
    """The DLL did not return within the timeout and the worker was killed."""

    def __init__(self, what: str, timeout: float) -> None:
        super().__init__(
            f"{what} did not return within {timeout:.0f} s.\n\n"
            + VISA_HANG_HELP
        )
        self.what = what
        self.timeout = timeout


# --------------------------------------------------------------- the worker


def _worker_main(cmd_q, res_q, dll_path: Optional[str]) -> None:
    """Child process: own the DLL, execute commands, post results back."""
    try:
        from .wgfmu_dll import Wgfmu

        drv = Wgfmu(dll_path)
    except BaseException as exc:  # noqa: BLE001 - must reach the parent
        res_q.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    res_q.put(("ok", None))

    while True:
        try:
            op, args = cmd_q.get()
        except (EOFError, OSError):
            break
        if op == "quit":
            try:
                drv.close()
            except Exception:
                pass
            return
        try:
            if op == "open":
                drv.open(args["address"])
                result = None
            elif op == "channels":
                result = drv.channels()
            elif op == "run":
                # Logging cannot cross the process boundary as a callback, so
                # collect lines and return them with the data.
                lines: List[str] = []
                data = drv.run(args["plan"], args["hw"], lines.append)
                result = (data, lines)
            elif op == "close":
                drv.close()
                result = None
            elif op == "ping":
                result = "pong"
            else:
                raise DriverError(f"unknown command {op!r}")
            res_q.put(("ok", result))
        except BaseException as exc:  # noqa: BLE001
            res_q.put(
                (
                    "error",
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
            )


# --------------------------------------------------------------- the parent


class RemoteWgfmu(Driver):
    """WGFMU driver that runs ``wgfmu.dll`` in a killable child process."""

    def __init__(
        self,
        dll_path: Optional[str] = None,
        open_timeout: float = OPEN_TIMEOUT,
        call_timeout: float = CALL_TIMEOUT,
    ) -> None:
        self._dll_path = dll_path or DEFAULT_DLL
        self._open_timeout = open_timeout
        self._call_timeout = call_timeout
        self._proc: Optional[mp.process.BaseProcess] = None
        self._cmd_q = None
        self._res_q = None
        self._open = False

    @property
    def name(self) -> str:
        return "WGFMU"

    @property
    def is_open(self) -> bool:
        return self._open

    # ----------------------------------------------------------- lifecycle

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        ctx = mp.get_context("spawn")
        self._cmd_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(self._cmd_q, self._res_q, self._dll_path),
            daemon=True,
        )
        self._proc.start()
        # The worker reports whether the DLL loaded at all.
        try:
            status = self._res_q.get(timeout=self._open_timeout)
        except queue.Empty:
            self._kill()
            raise WgfmuHang("loading wgfmu.dll", self._open_timeout) from None
        if status[0] != "ok":
            self._kill()
            raise WgfmuError(status[1])

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.join(timeout=5.0)
            except Exception:
                pass
        self._proc = None
        self._cmd_q = None
        self._res_q = None
        self._open = False

    def _call(self, op: str, timeout: float, **args):
        self._ensure_worker()
        self._cmd_q.put((op, args))
        try:
            reply = self._res_q.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise WgfmuHang(f"WGFMU {op}", timeout) from None
        if reply[0] != "ok":
            raise WgfmuError(reply[1])
        return reply[1]

    # -------------------------------------------------------------- Driver

    def open(self, address: str = DEFAULT_ADDRESS) -> None:
        self._call("open", self._open_timeout, address=address)
        self._open = True

    def close(self) -> None:
        if self._proc is None:
            self._open = False
            return
        try:
            self._call("close", CLOSE_TIMEOUT)
        except DriverError:
            pass  # a hung close still gets the process killed below
        finally:
            self._open = False
            try:
                if self._proc is not None and self._proc.is_alive():
                    self._cmd_q.put(("quit", {}))
                    self._proc.join(timeout=5.0)
            except Exception:
                pass
            if self._proc is not None and self._proc.is_alive():
                self._kill()
            else:
                self._proc = None

    def channels(self) -> List[int]:
        return self._call("channels", self._call_timeout)

    def run(self, plan: Plan, hw: Hardware, log: LogFn = None) -> Results:
        # A real run legitimately takes as long as the waveform, plus margin.
        timeout = plan.duration() * 2.0 + self._call_timeout + 60.0
        data, lines = self._call("run", timeout, plan=plan, hw=hw)
        if log:
            for line in lines:
                log(line)
        return data

    def abort(self) -> None:
        """Abort by killing the worker; the DLL offers no safe async abort."""
        self._kill()
