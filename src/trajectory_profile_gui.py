import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np

from trajectory_profile import TrajectoryProfile

class TrajectoryGUI:
    """A graphical user interface for generating and visualizing trajectory profiles."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        print("Initializing GUI")
        self.root = root
        self.root.title("Trajectory Profile GUI")

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create input fields
        self.create_widgets()

    def create_widgets(self):
        """Create input widgets."""
        # Left frame for inputs and buttons
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right frame for plots
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Input labels and entries with default values
        self.entries = {}
        self.params = ['Sampling Time', 'Position Start', 'Position End', 'Velocity', 'Acceleration', 'Jerk']
        self.defaults = {
            'Sampling Time': '0.004',
            'Position Start': '0.0',
            'Position End': '10.0',
            'Velocity': '10.0',
            'Acceleration': '100.0',
            'Jerk': '1000.0'
        }
        
        for i, param in enumerate(self.params):
            label = ttk.Label(self.left_frame, text=param)
            label.grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = ttk.Entry(self.left_frame)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            # Set default value
            entry.insert(0, self.defaults[param])
            self.entries[param] = entry

        # Buttons
        self.plot_button = ttk.Button(self.left_frame, text="Plot Trajectory", command=self.plot_trajectory)
        self.plot_button.grid(row=len(self.params), column=0, padx=5, pady=5)
        
        self.save_button = ttk.Button(self.left_frame, text="Save to CSV", command=self.save_to_csv)
        self.save_button.grid(row=len(self.params), column=1, padx=5, pady=5)

        self.clear_button = ttk.Button(self.left_frame, text="Clear All", command=self.clear_fields)
        self.clear_button.grid(row=len(self.params) + 1, column=0, padx=5, pady=5)

        self.reset_button = ttk.Button(self.left_frame, text="Reset Defaults", command=self.reset_defaults)
        self.reset_button.grid(row=len(self.params) + 1, column=1, padx=5, pady=5)

        # Matplotlib figure with navigation toolbar
        self.fig, self.ax = plt.subplots(4, 1, figsize=(12, 10))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar for zooming and panning
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)
        self.toolbar.update()

    def plot_trajectory(self):
        """Plot the trajectory based on user input."""
        try:
            # Get input values and validate
            sampling_time = self.validate_input('Sampling Time')
            pos_start = self.validate_input('Position Start')
            pos_end = self.validate_input('Position End')
            velocity = self.validate_input('Velocity')
            acceleration = self.validate_input('Acceleration')
            jerk = self.validate_input('Jerk')

            # Create a TrajectoryProfile instance
            tp = TrajectoryProfile()
            filtered_position_array = tp.filter(sampling_time, pos_start, pos_end, velocity, acceleration, jerk)

            if len(filtered_position_array) == 0:
                raise ValueError("Filtered position array is empty.")

            # Calculate derivatives for filtered arrays
            filtered_velocity_array = np.diff(filtered_position_array) / sampling_time
            filtered_acceleration_array = np.diff(filtered_velocity_array) / sampling_time
            filtered_jerk_array = np.diff(filtered_acceleration_array) / sampling_time  

            # Calculate derivatives for original arrays
            direction = np.sign(pos_end - pos_start)
            position_array = np.arange(pos_start, pos_end, velocity * direction * sampling_time)
            velocity_array = np.diff(position_array) / sampling_time
            acceleration_array = np.diff(velocity_array) / sampling_time
            jerk_array = np.diff(acceleration_array) / sampling_time

            # Create time arrays for proper x-axis plotting
            time_pos = np.arange(len(filtered_position_array)) * sampling_time
            time_vel = np.arange(len(filtered_velocity_array)) * sampling_time
            time_acc = np.arange(len(filtered_acceleration_array)) * sampling_time
            time_jerk = np.arange(len(filtered_jerk_array)) * sampling_time
            
            time_pos_orig = np.arange(len(position_array)) * sampling_time
            time_vel_orig = np.arange(len(velocity_array)) * sampling_time
            time_acc_orig = np.arange(len(acceleration_array)) * sampling_time
            time_jerk_orig = np.arange(len(jerk_array)) * sampling_time

            # Clear previous plots
            for ax in self.ax:
                ax.clear()

            # Plot the trajectory with time on x-axis
            self.ax[0].plot(time_pos_orig, position_array, "k", label="original")
            self.ax[0].plot(time_pos, filtered_position_array, "b-.", label="filtered")
            self.ax[0].set_ylabel("Position [deg]")
            self.ax[0].set_title("Position, Velocity, Acceleration and Jerk Profile")
            self.ax[0].legend()
            self.ax[0].grid()

            self.ax[1].plot(time_vel_orig, velocity_array, "k", label="original")
            self.ax[1].plot(time_vel, filtered_velocity_array, "b-.", label="filtered")
            self.ax[1].set_ylabel("Velocity [deg/s]")
            self.ax[1].legend()
            self.ax[1].grid()

            self.ax[2].plot(time_acc_orig, acceleration_array, "k", label="original")
            self.ax[2].plot(time_acc, filtered_acceleration_array, "b-.", label="filtered")
            self.ax[2].set_ylabel("Acceleration [deg/s^2]")
            self.ax[2].legend()
            self.ax[2].grid()
            
            self.ax[3].plot(time_jerk_orig, jerk_array, "k", label="original")
            self.ax[3].plot(time_jerk, filtered_jerk_array, "b-.", label="filtered")
            self.ax[3].set_ylabel("Jerk [deg/s^3]")
            self.ax[3].set_xlabel("Time [s]")
            self.ax[3].legend()
            self.ax[3].grid()

            self.canvas.draw()

            min_length = min(len(filtered_position_array), len(filtered_velocity_array), 
                           len(filtered_acceleration_array), len(filtered_jerk_array))
            time = np.arange(min_length) * sampling_time
            self.trajectory_data = pd.DataFrame({
                'Time': time,
                'Position': filtered_position_array[:min_length],
                'Velocity': filtered_velocity_array[:min_length],
                'Acceleration': filtered_acceleration_array[:min_length],
                'Jerk': filtered_jerk_array[:min_length]
            })

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def validate_input(self, param):
        """Validate and convert input to float."""
        value = self.entries[param].get()
        try:
            float_value = float(value)
            
            # Check for negative values in velocity, acceleration, and jerk
            if param in ['Velocity', 'Acceleration', 'Jerk'] and float_value < 0:
                raise ValueError(f"{param} must be positive. The sign will be calculated automatically based on the direction of motion (Position End - Position Start). Please enter a positive number.")
            
            return float_value
        except ValueError as e:
            if "could not convert" in str(e) or "invalid literal" in str(e):
                raise ValueError(f"Invalid value for {param}: {value}")
            else:
                # Re-raise our custom error message
                raise e

    def save_to_csv(self):
        """Save trajectory data to a CSV file."""
        if hasattr(self, 'trajectory_data'):
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.trajectory_data.to_csv(file_path, index=False)
                messagebox.showinfo("Save Successful", f"Trajectory saved to {file_path}")
        else:
            messagebox.showerror("No Data", "No trajectory data to save. Please plot the trajectory first.")

    def clear_fields(self):
        """Clear all input fields."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)

    def reset_defaults(self):
        """Reset all fields to default values."""
        for param, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, self.defaults[param])

if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryGUI(root)
    root.mainloop()
