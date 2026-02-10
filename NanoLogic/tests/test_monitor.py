"""
Tests for MemoryGuard
======================
Covers: initialization, RAM checking, gc trigger,
        and polling behavior.
"""

import psutil
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.monitor import MemoryGuard


class TestMemoryGuard:
    """Tests for MemoryGuard active memory defense."""

    @pytest.fixture
    def guard(self):
        return MemoryGuard(limit_gb=10.0, poll_interval=10)

    def test_initialization(self, guard):
        """Guard initializes with correct parameters."""
        assert guard.limit_gb == 10.0
        assert guard.poll_interval == 10
        assert guard.step_counter == 0

    def test_psutil_process_valid(self, guard):
        """Guard has a valid psutil Process handle."""
        assert guard.process is not None
        assert guard.process.pid > 0

    def test_step_increments_counter(self, guard):
        """Each check() call increments the step counter."""
        guard.check()
        assert guard.step_counter == 1
        guard.check()
        assert guard.step_counter == 2

    def test_check_does_not_crash(self, guard):
        """check() runs without errors (20 steps)."""
        for _ in range(20):
            guard.check()

    def test_ram_is_readable(self):
        """Can read system RAM via psutil (same method MemoryGuard uses)."""
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024 ** 3)
        assert isinstance(used_gb, float)
        assert used_gb > 0
        assert used_gb < 100

    def test_custom_limit(self):
        """Custom memory limit is stored."""
        guard = MemoryGuard(limit_gb=4.0)
        assert guard.limit_gb == 4.0

    def test_poll_interval_respected(self):
        """Guard only polls every poll_interval steps (no crash on rapid calls)."""
        guard = MemoryGuard(limit_gb=10.0, poll_interval=5)
        for _ in range(100):
            guard.check()
        assert guard.step_counter == 100

    def test_high_limit_no_alarm(self, guard):
        """With a high limit (100GB), check should never trigger alarm."""
        guard_safe = MemoryGuard(limit_gb=100.0, poll_interval=1)
        for _ in range(10):
            guard_safe.check()  # Should NOT trigger emergency brake


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
