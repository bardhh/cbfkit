import collections
import threading
from typing import DefaultDict, Dict

class JitMonitor:
    """Tracks JIT compilation events across the library."""
    _counts: DefaultDict[str, int] = collections.defaultdict(int)
    _lock = threading.Lock()

    @classmethod
    def increment(cls, key: str) -> None:
        """Increment the compilation count for a given key (function name)."""
        with cls._lock:
            cls._counts[key] += 1

    @classmethod
    def get_counts(cls) -> Dict[str, int]:
        """Return a copy of the current compilation counts."""
        with cls._lock:
            return dict(cls._counts)

    @classmethod
    def reset(cls) -> None:
        """Reset all counts to zero."""
        with cls._lock:
            cls._counts.clear()
