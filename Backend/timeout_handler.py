# =====================================================================================
# TIMEOUT HANDLER CLASS
# =====================================================================================
import threading


class TimeoutException(Exception):
    """Exception raised when model loading times out"""
    pass


class ModelLoadingTimeout:
    """Context manager for handling model loading timeouts with proper cancellation"""

    def __init__(self, timeout_seconds=120):  # 2 minutes
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.cancelled = False

    def _timeout_handler(self):
        if not self.cancelled:
            self.cancelled = True
            raise TimeoutException(f"Model loading timed out after {self.timeout_seconds} seconds")

    def __enter__(self):
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cancelled = True
        if self.timer:
            self.timer.cancel()

    def is_cancelled(self):
        return self.cancelled
