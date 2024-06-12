import unittest
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Get the absolute path of the project directory and append the src directory
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_directory, 'src')
sys.path.append(src_path)
from trajectory_profile import TrajectoryProfile


class TestTrajectoryProfile(unittest.TestCase):


    def test_trajectory_profile_1(self):
        trajectory_profile = TrajectoryProfile()
        sampling_time= 0.001 
        position_start= 0 
        position_end= 50
        velocity= 50 
        acceleration= 500
        filtered_position_array = trajectory_profile.filter(sampling_time= sampling_time, position_start= position_start, position_end= position_end, velocity= velocity, acceleration= acceleration)
        filtered_velocity_array = np.diff(filtered_position_array) / sampling_time
        filtered_acceleration_array = np.diff(filtered_velocity_array) / sampling_time

        position_array = np.arange(position_start, position_end, velocity * sampling_time)
        velocity_array = np.diff(position_array) / sampling_time
        acceleration_array = np.diff(velocity_array) / sampling_time

        plt.figure(figsize=(10, 8))
        plt.subplot(3,1,1)
        plt.plot(position_array, "k",label= "original")
        plt.plot(filtered_position_array, "b-.",label= "filtered")
        plt.xlabel("Sample")
        plt.ylabel("Position [deg]")
        plt.title("Position, Velocity and Acceleratiom Profile")
        plt.legend()
        plt.grid()

        plt.subplot(3,1,2)
        plt.plot(velocity_array, "k",label= "original")
        plt.plot(filtered_velocity_array, "b-.",label= "filtered")
        plt.xlabel("Sample")
        plt.ylabel("Position [deg]")
        plt.legend()
        plt.grid()

        plt.subplot(3,1,3)
        plt.plot(filtered_acceleration_array, "k", label= "original")
        plt.plot(acceleration_array, "b-.", label= "filtered")
        plt.xlabel("Sample")
        plt.ylabel("Position [deg]")
        plt.legend()
        plt.grid()

        plt.show()
        