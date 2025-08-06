import numpy as np


class TrajectoryProfile:
    "Trajectory Profile Class"

    def __init__(self) -> None:
        pass

    def filter(
        self,
        sampling_time: float,
        position_start: float,
        position_end: float,
        velocity: float,
        acceleration: float,
        jerk: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            sampling_time (float)
            position_start (float)
            position_end (float)
            velocity (float)
            acceleration (float)
            jerk (float)

        Raises:
            ValueError: FIR Filter window size is zero

        Returns:
            np.ndarray: position trajectory profile
        """

        range_motion = np.abs(position_end - position_start)
        fir_filter_time_constant = np.abs(velocity / acceleration)
        jerk_filter_time_constant = np.abs(acceleration / jerk)
        direction = np.sign(position_end - position_start)

        if np.abs(range_motion / velocity) < fir_filter_time_constant:
            velocity = range_motion / fir_filter_time_constant
        duration = np.abs(range_motion / velocity)

        t_array = np.arange(0, duration, sampling_time)
        vel_array = np.ones_like(t_array) * velocity * direction

        # Check for zero window sizes
        if int((fir_filter_time_constant) / sampling_time) == 0:
            raise ValueError("FIR Filter window size is zero")
        if int((jerk_filter_time_constant) / sampling_time) == 0:
            raise ValueError("Jerk FIR Filter window size is zero")

        # Jerk filter
        jerk_filter = (
            np.ones(int((jerk_filter_time_constant) / sampling_time))
            * 1
            / (jerk_filter_time_constant)
        )

        # Acceleration filter
        acc_filter = (
            np.ones(int((fir_filter_time_constant) / sampling_time))
            * 1
            / (fir_filter_time_constant)
        )

        # apply acceleration filter
        acc_filtered_velocity = (
            np.convolve(vel_array, acc_filter, mode="full") * sampling_time
        )

        # apply jerk filter
        jerk_filtered_velocity = (
            np.convolve(acc_filtered_velocity, jerk_filter, mode="full") * sampling_time
        )

        filtered_velocity = np.concatenate([[0], jerk_filtered_velocity, [0]])
        filtered_position = (
            np.cumsum(filtered_velocity * sampling_time) + position_start
        )

        return filtered_position


class PredefinedTrajectoryProfile:
    def __init__(
        self,
        type: str,
        position_start: float,
        sampling_time: float,
        duration: float,
        amplitude: float,
        frequency: float = 1.0,
        steepness: float = 1.0,
        center_time: float = None,
    ):
        self.type = type
        self.position_start = position_start
        self.sampling_time = sampling_time
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.steepness = steepness
        self.center_time = center_time if center_time is not None else duration / 2

        self.time = np.arange(0, duration, sampling_time)

        if type == "smooth_sine":
            self.trajectory = self._generate_sigmoid_sine_trajectory()
        else:
            raise ValueError(f"Invalid trajectory profile type: {type}")

    def _smooth_sigmoid(self, t, steepness=1.0, center=0.0):
        """Improved sigmoid function with controllable steepness and center."""
        t_shifted = steepness * (t - center)
        return np.exp(-np.exp(-t_shifted))

    def _generate_sigmoid_sine_trajectory(self):
        """Generate sigmoid-sine product trajectory with improved parameters."""
        t_normalized = self.steepness * (self.time - self.center_time)

        sigmoid_envelope = self._smooth_sigmoid(
            self.time, self.steepness, self.center_time
        )

        sine_wave = np.sin(2 * np.pi * self.frequency * self.time)

        trajectory = self.position_start + self.amplitude * sigmoid_envelope * sine_wave

        return trajectory

    def get_trajectory(self):
        """Get the position trajectory."""
        return self.trajectory

    def get_trajectory_stats(self):
        """Get statistical information about the trajectory."""
        velocity, acceleration, jerk = self.get_derivatives()

        stats = {
            "duration": self.duration,
            "position_range": (np.min(self.trajectory), np.max(self.trajectory)),
            "max_velocity": np.max(np.abs(velocity)),
            "max_acceleration": np.max(np.abs(acceleration)),
            "max_jerk": np.max(np.abs(jerk)),
            "start_position": self.trajectory[0],
            "end_position": self.trajectory[-1],
            "start_velocity": velocity[0],
            "end_velocity": velocity[-1],
        }

        return stats

