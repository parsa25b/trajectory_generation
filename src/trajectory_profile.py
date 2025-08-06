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
