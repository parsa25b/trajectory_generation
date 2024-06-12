import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

        # Create input fields
        self.create_widgets()

    def create_widgets(self):
        """Create input widgets."""
        # Input labels and entries
        self.entries = {}
        self.params = ['Sampling Time', 'Position Start', 'Position End', 'Velocity', 'Acceleration']
        
        for i, param in enumerate(self.params):
            label = ttk.Label(self.root, text=param)
            label.grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = ttk.Entry(self.root)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            self.entries[param] = entry

        # Buttons
        self.plot_button = ttk.Button(self.root, text="Plot Trajectory", command=self.plot_trajectory)
        self.plot_button.grid(row=len(self.params), column=0, padx=5, pady=5)
        
        self.save_button = ttk.Button(self.root, text="Save to CSV", command=self.save_to_csv)
        self.save_button.grid(row=len(self.params), column=1, padx=5, pady=5)

        self.clear_button = ttk.Button(self.root, text="Clear All", command=self.clear_fields)
        self.clear_button.grid(row=len(self.params) + 1, column=0, padx=5, pady=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=len(self.params) + 2, column=0, columnspan=2)

    def plot_trajectory(self):
        """Plot the trajectory based on user input."""
        try:
            # Get input values and validate
            sampling_time = self.validate_input('Sampling Time')
            pos_start = self.validate_input('Position Start')
            pos_end = self.validate_input('Position End')
            velocity = self.validate_input('Velocity')
            acceleration = self.validate_input('Acceleration')

            # Create a TrajectoryProfile instance
            tp = TrajectoryProfile()
            filtered_position_array = tp.filter(sampling_time, pos_start, pos_end, velocity, acceleration)

            if len(filtered_position_array) == 0:
                raise ValueError("Filtered position array is empty.")

            filtered_velocity_array = np.gradient(filtered_position_array, sampling_time)
            filtered_acceleration_array = np.gradient(filtered_velocity_array, sampling_time)

            position_array = np.arange(pos_start, pos_end, velocity * sampling_time)
            velocity_array = np.diff(position_array) / sampling_time
            acceleration_array = np.diff(velocity_array) / sampling_time

            # Clear previous plots
            for ax in self.ax:
                ax.clear()

            # Plot the trajectory
            self.ax[0].plot(position_array, "k", label="original")
            self.ax[0].plot(filtered_position_array, "b-.", label="filtered")
            self.ax[0].set_ylabel("Position [deg]")
            self.ax[0].set_title("Position, Velocity and Acceleration Profile")
            self.ax[0].legend()
            self.ax[0].grid()

            self.ax[1].plot(velocity_array, "k", label="original")
            self.ax[1].plot(filtered_velocity_array, "b-.", label="filtered")
            self.ax[1].set_ylabel("Velocity [deg/s]")
            self.ax[1].legend()
            self.ax[1].grid()

            self.ax[2].plot(acceleration_array, "k", label="original")
            self.ax[2].plot(filtered_acceleration_array, "b-.", label="filtered")
            self.ax[2].set_ylabel("Acceleration [deg/s^2]")
            self.ax[2].legend()
            self.ax[2].grid()

            self.canvas.draw()

            # Store the result for saving
            time = np.arange(0, len(filtered_position_array) * sampling_time, sampling_time)
            self.trajectory_data = pd.DataFrame({
                'Time': time,
                'Position': filtered_position_array,
                'Velocity': filtered_velocity_array,
                'Acceleration': filtered_acceleration_array
            })

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def validate_input(self, param):
        """Validate and convert input to float."""
        value = self.entries[param].get()
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid value for {param}: {value}")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryGUI(root)
    root.mainloop()
