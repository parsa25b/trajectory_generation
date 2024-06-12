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
    ) -> np.ndarray:
        """_summary_

        Args:
            sampling_time (float)
            position_start (float)
            position_end (float)
            velocity (float)
            acceleration (float)

        Raises:
            ValueError: FIR Filter window size is zero

        Returns:
            np.ndarray: position trajectory profile
        """        
        
        range_motion = position_end - position_start
        fir_filter_time_constant = velocity / acceleration
        if (range_motion / velocity) < fir_filter_time_constant:
            velocity = range_motion / velocity
        duration = range_motion / velocity

        t_array = np.arange(0, duration, sampling_time)
        vel_array = np.ones_like(t_array) * velocity

        if int((fir_filter_time_constant) / sampling_time) == 0:
            raise ValueError("FIR Filter window size is zero")
        fir_filter = (
            np.ones(int((fir_filter_time_constant) / sampling_time))
            * 1
            / (fir_filter_time_constant)
        )

        # velocity filter
        filtered_velocity = (
            np.convolve(vel_array, fir_filter, mode="full") * sampling_time
        )
        filtered_velocity = np.concatenate([[0], filtered_velocity, [0]])
        filtered_position = (
            np.cumsum(filtered_velocity * sampling_time) + position_start
        )

        return filtered_position
