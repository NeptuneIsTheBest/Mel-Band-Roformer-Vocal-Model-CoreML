from __future__ import annotations

import contextlib
import ctypes
import sys
import threading
import time

import psutil


class MacOSAutoreleasePool:
    def __init__(self) -> None:
        self._pool: int | None = None
        self._msg_send = None
        self._drain = 0

    def __enter__(self) -> "MacOSAutoreleasePool":
        if sys.platform != "darwin":
            return self
        try:
            ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
            objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            msg_send = objc.objc_msgSend
            msg_send.restype = ctypes.c_void_p
            msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            pool_class = objc.objc_getClass(b"NSAutoreleasePool")
            alloc = objc.sel_registerName(b"alloc")
            init = objc.sel_registerName(b"init")
            drain = objc.sel_registerName(b"drain")
            self._pool = msg_send(msg_send(pool_class, alloc), init)
            self._msg_send = msg_send
            self._drain = drain
        except Exception:  # noqa: BLE001
            self._pool = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._pool is not None and self._msg_send is not None:
            self._msg_send(self._pool, self._drain)
        self._pool = None


def autorelease_pool() -> contextlib.AbstractContextManager[object]:
    if sys.platform == "darwin":
        return MacOSAutoreleasePool()
    return contextlib.nullcontext()


class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.start_rss_mb = self._rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self.end_rss_mb = self.start_rss_mb
        self._running = False
        self._thread: threading.Thread | None = None

    def _rss_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

    def _sample(self) -> None:
        while self._running:
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
            time.sleep(self.interval)

    def __enter__(self) -> "PeakMemoryMonitor":
        self.start_rss_mb = self._rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_rss_mb = self._rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, self.end_rss_mb)

    def result(self, prefix: str | None = None) -> dict[str, float]:
        values = {
            "rss_start_mb": self.start_rss_mb,
            "rss_peak_mb": self.peak_rss_mb,
            "rss_end_mb": self.end_rss_mb,
        }
        if prefix is None:
            return values
        return {f"{prefix}_{key}": value for key, value in values.items()}


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024
