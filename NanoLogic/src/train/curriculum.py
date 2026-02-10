class CurriculumScheduler:
    """
    Manages the difficulty of the training data.
    
    Rounds double each phase: 8 → 16 → 32 → 64 (configurable).
    Phase promotion requires BOTH:
        1. Minimum steps completed in the current phase.
        2. Running accuracy >= the phase's accuracy threshold.
    
    The starting phase is determined by config.start_round, regardless
    of any checkpoint — so training always begins at the configured round.
    """
    def __init__(self, config):
        self.rounds_schedule = list(config.curriculum_rounds)
        self.phase_min_steps = list(config.phase_min_steps)
        self.phase_accuracy_thresholds = list(config.phase_accuracy_thresholds)
        
        # Determine starting phase from config.start_round
        start_round = config.start_round
        self.current_phase = 0
        for i, r in enumerate(self.rounds_schedule):
            if r == start_round:
                self.current_phase = i
                break
        
        self.current_step = 0
        self.total_steps = 0
        self._running_acc_sum = 0.0
        self._running_acc_count = 0
        
    def step(self, accuracy=None):
        """
        Advances the scheduler by one step.
        
        Args:
            accuracy: Current step's bit-level accuracy (0.0–1.0).
                      Required for accuracy-gated promotion.
        
        Returns:
            Tuple(rounds, phase_changed_bool)
        """
        self.current_step += 1
        self.total_steps += 1
        
        # Track running accuracy for promotion decision
        if accuracy is not None:
            self._running_acc_sum += accuracy
            self._running_acc_count += 1
        
        # Check promotion conditions
        if self.current_phase < len(self.rounds_schedule) - 1:
            min_steps_met = self.current_step >= self.phase_min_steps[self.current_phase]
            
            # Compute running average accuracy
            if self._running_acc_count > 0:
                avg_acc = self._running_acc_sum / self._running_acc_count
            else:
                avg_acc = 0.0
            
            threshold = self.phase_accuracy_thresholds[self.current_phase]
            accuracy_met = avg_acc >= threshold
            
            if min_steps_met and accuracy_met:
                self.current_phase += 1
                self.current_step = 0
                self._running_acc_sum = 0.0
                self._running_acc_count = 0
                return self.get_current_rounds(), True
        
        return self.get_current_rounds(), False
        
    def get_current_rounds(self):
        return self.rounds_schedule[self.current_phase]
    
    def get_accuracy_threshold(self):
        """Returns the accuracy threshold for the current phase."""
        return self.phase_accuracy_thresholds[self.current_phase]
    
    def get_running_accuracy(self):
        """Returns the running average accuracy in the current phase."""
        if self._running_acc_count > 0:
            return self._running_acc_sum / self._running_acc_count
        return 0.0
    
    def state_dict(self):
        return {
            'current_phase': self.current_phase,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'running_acc_sum': self._running_acc_sum,
            'running_acc_count': self._running_acc_count,
        }
        
    def load_state_dict(self, state_dict):
        self.current_phase = state_dict['current_phase']
        self.current_step = state_dict['current_step']
        self.total_steps = state_dict['total_steps']
        self._running_acc_sum = state_dict.get('running_acc_sum', 0.0)
        self._running_acc_count = state_dict.get('running_acc_count', 0)
