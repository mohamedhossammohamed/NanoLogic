"""
Tests for Curriculum Scheduler
================================
Covers: accuracy-gated promotion, configurable thresholds,
        start_round override, step counting, and state dict save/load.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.train.curriculum import CurriculumScheduler


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    @pytest.fixture
    def scheduler(self):
        return CurriculumScheduler(Config())

    def test_initial_phase(self, scheduler):
        """Starts at Phase 0 (8 rounds) by default."""
        assert scheduler.current_phase == 0
        assert scheduler.get_current_rounds() == 8

    def test_no_promotion_without_accuracy(self, scheduler):
        """Does NOT promote even after min_steps if accuracy is low."""
        for _ in range(1000):
            scheduler.step(accuracy=0.5)  # 50% — below 95% threshold
        assert scheduler.current_phase == 0, "Should NOT have promoted with 50% accuracy"
        assert scheduler.get_current_rounds() == 8

    def test_no_promotion_before_min_steps(self, scheduler):
        """Does NOT promote before min_steps even if accuracy is 100%."""
        for _ in range(499):
            scheduler.step(accuracy=1.0)
        assert scheduler.current_phase == 0, "Should NOT promote before 500 min_steps"

    def test_promotion_with_accuracy_and_min_steps(self, scheduler):
        """Promotes after min_steps AND accuracy >= threshold."""
        for _ in range(500):
            scheduler.step(accuracy=0.96)
        assert scheduler.current_phase == 1, "Should promote after 500 steps with 96% accuracy"
        assert scheduler.get_current_rounds() == 16

    def test_phase_2_requires_accuracy(self, scheduler):
        """Phase 1 → Phase 2 also requires accuracy gate."""
        # Pass phase 0
        for _ in range(500):
            scheduler.step(accuracy=0.96)
        assert scheduler.current_phase == 1

        # Stay in phase 1 with low accuracy
        for _ in range(5000):
            scheduler.step(accuracy=0.80)
        assert scheduler.current_phase == 1, "Should NOT promote from phase 1 with 80% accuracy"

    def test_full_promotion_chain(self, scheduler):
        """Can promote through all phases with high accuracy."""
        # Phase 0 → 1 (500 steps, 96%)
        for _ in range(500):
            scheduler.step(accuracy=0.96)
        assert scheduler.current_phase == 1

        # Phase 1 → 2 (2000 steps, 96%)
        for _ in range(2000):
            scheduler.step(accuracy=0.96)
        assert scheduler.current_phase == 2

        # Phase 2 → 3 (5000 steps, 96%)
        for _ in range(5000):
            scheduler.step(accuracy=0.96)
        assert scheduler.current_phase == 3

    def test_last_phase_never_promotes(self, scheduler):
        """Phase 3 (64 rounds) runs forever."""
        # Fast-forward to phase 3
        for _ in range(500):
            scheduler.step(accuracy=0.96)
        for _ in range(2000):
            scheduler.step(accuracy=0.96)
        for _ in range(5000):
            scheduler.step(accuracy=0.96)
        assert scheduler.current_phase == 3
        
        # Even with perfect accuracy, stays at phase 3
        for _ in range(10000):
            scheduler.step(accuracy=1.0)
        assert scheduler.current_phase == 3

    def test_step_counting(self, scheduler):
        """Total steps are counted correctly."""
        for _ in range(100):
            scheduler.step(accuracy=0.5)
        assert scheduler.total_steps == 100

    def test_state_dict_roundtrip(self, scheduler):
        """state_dict → load roundtrip preserves state."""
        for _ in range(300):
            scheduler.step(accuracy=0.7)

        state = scheduler.state_dict()
        new_scheduler = CurriculumScheduler(Config())
        new_scheduler.load_state_dict(state)

        assert new_scheduler.current_phase == scheduler.current_phase
        assert new_scheduler.total_steps == scheduler.total_steps
        assert new_scheduler.get_current_rounds() == scheduler.get_current_rounds()
        assert abs(new_scheduler.get_running_accuracy() - scheduler.get_running_accuracy()) < 1e-6

    def test_four_phases_exist(self, scheduler):
        """Schedule has exactly 4 phases: 8, 16, 32, 64."""
        assert len(scheduler.rounds_schedule) == 4
        assert scheduler.rounds_schedule == [8, 16, 32, 64]

    def test_configurable_thresholds(self):
        """Custom accuracy thresholds are respected."""
        config = Config()
        config.phase_accuracy_thresholds = [0.80, 0.85, 0.90, 1.0]
        scheduler = CurriculumScheduler(config)
        
        # 82% should pass an 80% gate after min_steps
        for _ in range(500):
            scheduler.step(accuracy=0.82)
        assert scheduler.current_phase == 1

    def test_start_round_override(self):
        """config.start_round controls the starting phase."""
        config = Config()
        config.start_round = 32
        scheduler = CurriculumScheduler(config)
        
        assert scheduler.current_phase == 2
        assert scheduler.get_current_rounds() == 32

    def test_start_round_16(self):
        """start_round=16 starts at phase 1."""
        config = Config()
        config.start_round = 16
        scheduler = CurriculumScheduler(config)
        
        assert scheduler.current_phase == 1
        assert scheduler.get_current_rounds() == 16

    def test_get_accuracy_threshold(self, scheduler):
        """Returns the threshold for the current phase."""
        assert scheduler.get_accuracy_threshold() == 0.95

    def test_running_accuracy(self, scheduler):
        """Running accuracy is tracked correctly."""
        for _ in range(10):
            scheduler.step(accuracy=0.80)
        assert abs(scheduler.get_running_accuracy() - 0.80) < 1e-6

    def test_running_accuracy_resets_on_promotion(self, scheduler):
        """Running accuracy resets when phase changes."""
        for _ in range(500):
            scheduler.step(accuracy=0.96)
        # Just promoted — running accuracy should be reset
        assert scheduler.get_running_accuracy() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
