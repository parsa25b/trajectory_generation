import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np

from trajectory_profile import TrajectoryProfile, PredefinedTrajectoryProfile


class TrajectoryGUI:
    """A graphical user interface for generating and visualizing trajectory profiles."""

    def __init__(self, root):
        """Initialize the GUI."""
        print("Initializing GUI")
        self.root = root
        self.root.title("Trajectory Profile GUI")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_tabs()

    def create_tabs(self):
        """Create tabbed interface."""
        # Tab 1: Custom Trajectory
        self.custom_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.custom_tab, text="Custom Trajectory")

        # Tab 2: Predefined Trajectory
        self.predefined_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predefined_tab, text="Predefined Trajectory")

        # Create widgets for each tab
        self.create_custom_widgets()
        self.create_predefined_widgets()

    def create_custom_widgets(self):
        """Create input widgets for custom trajectory tab."""
        # Left frame for inputs and buttons
        self.custom_left_frame = ttk.Frame(self.custom_tab)
        self.custom_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Right frame for plots
        self.custom_right_frame = ttk.Frame(self.custom_tab)
        self.custom_right_frame.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Input labels and entries with default values
        self.entries = {}
        self.params = [
            "Sampling Time",
            "Position Start",
            "Position End",
            "Velocity",
            "Acceleration",
            "Jerk",
        ]
        self.defaults = {
            "Sampling Time": "0.004",
            "Position Start": "0.0",
            "Position End": "10.0",
            "Velocity": "10.0",
            "Acceleration": "100.0",
            "Jerk": "1000.0",
        }

        for i, param in enumerate(self.params):
            label = ttk.Label(self.custom_left_frame, text=param)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(self.custom_left_frame)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            # Set default value
            entry.insert(0, self.defaults[param])
            self.entries[param] = entry

        # Buttons
        self.plot_button = ttk.Button(
            self.custom_left_frame, text="Plot Trajectory", command=self.plot_trajectory
        )
        self.plot_button.grid(row=len(self.params), column=0, padx=5, pady=5)

        self.save_button = ttk.Button(
            self.custom_left_frame, text="Save to CSV", command=self.save_to_csv
        )
        self.save_button.grid(row=len(self.params), column=1, padx=5, pady=5)

        self.clear_button = ttk.Button(
            self.custom_left_frame, text="Clear All", command=self.clear_fields
        )
        self.clear_button.grid(row=len(self.params) + 1, column=0, padx=5, pady=5)

        self.reset_button = ttk.Button(
            self.custom_left_frame, text="Reset Defaults", command=self.reset_defaults
        )
        self.reset_button.grid(row=len(self.params) + 1, column=1, padx=5, pady=5)

        # Matplotlib figure with navigation toolbar
        self.fig, self.ax = plt.subplots(4, 1, figsize=(12, 10))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.custom_right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for zooming and panning
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.custom_right_frame)
        self.toolbar.update()

    def create_predefined_widgets(self):
        """Create input widgets for predefined trajectory tab."""
        # Left frame for inputs and buttons
        self.predefined_left_frame = ttk.Frame(self.predefined_tab)
        self.predefined_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Right frame for plots
        self.predefined_right_frame = ttk.Frame(self.predefined_tab)
        self.predefined_right_frame.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Input labels and entries with default values for predefined trajectory
        self.predefined_entries = {}
        self.predefined_params = [
            "Sampling Time",
            "Position Start",
            "Duration",
            "Amplitude",
            "Frequency",
            "Steepness",
            "Center Time",
            "Type",
        ]
        self.predefined_defaults = {
            "Sampling Time": "0.004",
            "Position Start": "0.0",
            "Duration": "10.0",
            "Amplitude": "5.0",
            "Frequency": "1.0",
            "Steepness": "1.0",
            "Center Time": "5.0",
            "Type": "smooth_sine",
        }

        for i, param in enumerate(self.predefined_params):
            label = ttk.Label(self.predefined_left_frame, text=param)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="e")

            if param == "Type":
                # Create combobox for trajectory type
                combo = ttk.Combobox(
                    self.predefined_left_frame, values=["smooth_sine"], state="readonly"
                )
                combo.grid(row=i, column=1, padx=5, pady=5, sticky="w")
                combo.set(self.predefined_defaults[param])
                self.predefined_entries[param] = combo
            else:
                entry = ttk.Entry(self.predefined_left_frame)
                entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
                entry.insert(0, self.predefined_defaults[param])
                self.predefined_entries[param] = entry

        # Buttons for predefined trajectory
        self.predefined_plot_button = ttk.Button(
            self.predefined_left_frame,
            text="Plot Trajectory",
            command=self.plot_predefined_trajectory,
        )
        self.predefined_plot_button.grid(
            row=len(self.predefined_params), column=0, padx=5, pady=5
        )

        self.predefined_save_button = ttk.Button(
            self.predefined_left_frame,
            text="Save to CSV",
            command=self.save_predefined_to_csv,
        )
        self.predefined_save_button.grid(
            row=len(self.predefined_params), column=1, padx=5, pady=5
        )

        self.predefined_clear_button = ttk.Button(
            self.predefined_left_frame,
            text="Clear All",
            command=self.clear_predefined_fields,
        )
        self.predefined_clear_button.grid(
            row=len(self.predefined_params) + 1, column=0, padx=5, pady=5
        )

        self.predefined_reset_button = ttk.Button(
            self.predefined_left_frame,
            text="Reset Defaults",
            command=self.reset_predefined_defaults,
        )
        self.predefined_reset_button.grid(
            row=len(self.predefined_params) + 1, column=1, padx=5, pady=5
        )

        # Matplotlib figure with navigation toolbar for predefined trajectory
        self.predefined_fig, self.predefined_ax = plt.subplots(4, 1, figsize=(12, 10))
        self.predefined_fig.tight_layout(pad=3.0)

        self.predefined_canvas = FigureCanvasTkAgg(
            self.predefined_fig, master=self.predefined_right_frame
        )
        self.predefined_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for zooming and panning
        self.predefined_toolbar = NavigationToolbar2Tk(
            self.predefined_canvas, self.predefined_right_frame
        )
        self.predefined_toolbar.update()

    def plot_trajectory(self):
        """Plot the trajectory based on user input."""
        try:
            # Get input values and validate
            sampling_time = self.validate_input("Sampling Time")
            pos_start = self.validate_input("Position Start")
            pos_end = self.validate_input("Position End")
            velocity = self.validate_input("Velocity")
            acceleration = self.validate_input("Acceleration")
            jerk = self.validate_input("Jerk")

            # Create a TrajectoryProfile instance
            tp = TrajectoryProfile()
            filtered_position_array = tp.filter(
                sampling_time, pos_start, pos_end, velocity, acceleration, jerk
            )

            if len(filtered_position_array) == 0:
                raise ValueError("Filtered position array is empty.")

            # Calculate derivatives for filtered arrays
            filtered_velocity_array = np.diff(filtered_position_array) / sampling_time
            filtered_acceleration_array = (
                np.diff(filtered_velocity_array) / sampling_time
            )
            filtered_jerk_array = np.diff(filtered_acceleration_array) / sampling_time

            # Calculate derivatives for original arrays
            direction = np.sign(pos_end - pos_start)
            position_array = np.arange(
                pos_start, pos_end, velocity * direction * sampling_time
            )
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
            self.ax[2].plot(
                time_acc, filtered_acceleration_array, "b-.", label="filtered"
            )
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

            min_length = min(
                len(filtered_position_array),
                len(filtered_velocity_array),
                len(filtered_acceleration_array),
                len(filtered_jerk_array),
            )
            time = np.arange(min_length) * sampling_time
            self.trajectory_data = pd.DataFrame(
                {
                    "Time": time,
                    "Position": filtered_position_array[:min_length],
                    "Velocity": filtered_velocity_array[:min_length],
                    "Acceleration": filtered_acceleration_array[:min_length],
                    "Jerk": filtered_jerk_array[:min_length],
                }
            )

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
            if param in ["Velocity", "Acceleration", "Jerk"] and float_value < 0:
                raise ValueError(
                    f"{param} must be positive. The sign will be calculated automatically based on the direction of motion (Position End - Position Start). Please enter a positive number."
                )

            return float_value
        except ValueError as e:
            if "could not convert" in str(e) or "invalid literal" in str(e):
                raise ValueError(f"Invalid value for {param}: {value}")
            else:
                # Re-raise our custom error message
                raise e

    def save_to_csv(self):
        """Save trajectory data to a CSV file."""
        if hasattr(self, "trajectory_data"):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.trajectory_data.to_csv(file_path, index=False)
                messagebox.showinfo(
                    "Save Successful", f"Trajectory saved to {file_path}"
                )
        else:
            messagebox.showerror(
                "No Data",
                "No trajectory data to save. Please plot the trajectory first.",
            )

    def clear_fields(self):
        """Clear all input fields."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)

    def reset_defaults(self):
        """Reset all fields to default values."""
        for param, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, self.defaults[param])

    def plot_predefined_trajectory(self):
        """Plot the predefined trajectory based on user input."""
        try:
            # Get input values and validate
            sampling_time = self.validate_predefined_input("Sampling Time")
            position_start = self.validate_predefined_input("Position Start")
            duration = self.validate_predefined_input("Duration")
            amplitude = self.validate_predefined_input("Amplitude")
            frequency = self.validate_predefined_input("Frequency")
            steepness = self.validate_predefined_input("Steepness")
            center_time = self.validate_predefined_input("Center Time")
            trajectory_type = self.predefined_entries["Type"].get()

            # Create a PredefinedTrajectoryProfile instance
            predefined_tp = PredefinedTrajectoryProfile(
                type=trajectory_type,
                position_start=position_start,
                sampling_time=sampling_time,
                duration=duration,
                amplitude=amplitude,
                frequency=frequency,
                steepness=steepness,
                center_time=center_time,
            )

            # Get the trajectory
            position_array = predefined_tp.get_trajectory()
            time_array = predefined_tp.time

            if len(position_array) == 0:
                raise ValueError("Position array is empty.")

            # Calculate derivatives
            velocity_array = np.diff(position_array) / sampling_time
            acceleration_array = np.diff(velocity_array) / sampling_time
            jerk_array = np.diff(acceleration_array) / sampling_time

            # Create time arrays for proper x-axis plotting
            time_pos = time_array
            time_vel = time_array[:-1]
            time_acc = time_array[:-2]
            time_jerk = time_array[:-3]

            # Clear previous plots
            for ax in self.predefined_ax:
                ax.clear()

            # Plot the trajectory with time on x-axis
            self.predefined_ax[0].plot(
                time_pos, position_array, "b-", label="predefined"
            )
            self.predefined_ax[0].set_ylabel("Position [deg]")
            self.predefined_ax[0].set_title("Predefined Trajectory Profile")
            self.predefined_ax[0].legend()
            self.predefined_ax[0].grid()

            self.predefined_ax[1].plot(time_vel, velocity_array, "r-", label="velocity")
            self.predefined_ax[1].set_ylabel("Velocity [deg/s]")
            self.predefined_ax[1].legend()
            self.predefined_ax[1].grid()

            self.predefined_ax[2].plot(
                time_acc, acceleration_array, "g-", label="acceleration"
            )
            self.predefined_ax[2].set_ylabel("Acceleration [deg/s^2]")
            self.predefined_ax[2].legend()
            self.predefined_ax[2].grid()

            self.predefined_ax[3].plot(time_jerk, jerk_array, "m-", label="jerk")
            self.predefined_ax[3].set_ylabel("Jerk [deg/s^3]")
            self.predefined_ax[3].set_xlabel("Time [s]")
            self.predefined_ax[3].legend()
            self.predefined_ax[3].grid()

            self.predefined_canvas.draw()

            # Store data for saving
            min_length = min(
                len(position_array),
                len(velocity_array),
                len(acceleration_array),
                len(jerk_array),
            )
            time = time_array[:min_length]
            self.predefined_trajectory_data = pd.DataFrame(
                {
                    "Time": time,
                    "Position": position_array[:min_length],
                    "Velocity": velocity_array[:min_length],
                    "Acceleration": acceleration_array[:min_length],
                    "Jerk": jerk_array[:min_length],
                }
            )

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def validate_predefined_input(self, param):
        """Validate and convert predefined trajectory input to float."""
        value = self.predefined_entries[param].get()
        try:
            float_value = float(value)

            # Check for negative values where appropriate
            if (
                param
                in ["Sampling Time", "Duration", "Amplitude", "Frequency", "Steepness"]
                and float_value <= 0
            ):
                raise ValueError(f"{param} must be positive.")

            # Validate Center Time is within duration bounds
            if param == "Center Time":
                duration_entry = self.predefined_entries.get("Duration")
                if duration_entry:
                    try:
                        duration_val = float(duration_entry.get())
                        if float_value < 0 or float_value > duration_val:
                            raise ValueError(
                                f"Center Time must be between 0 and Duration ({duration_val})."
                            )
                    except ValueError:
                        pass  # If duration is invalid, let it be caught elsewhere

            return float_value
        except ValueError as e:
            if "could not convert" in str(e) or "invalid literal" in str(e):
                raise ValueError(f"Invalid value for {param}: {value}")
            else:
                # Re-raise our custom error message
                raise e

    def save_predefined_to_csv(self):
        """Save predefined trajectory data to a CSV file."""
        if hasattr(self, "predefined_trajectory_data"):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.predefined_trajectory_data.to_csv(file_path, index=False)
                messagebox.showinfo(
                    "Save Successful", f"Predefined trajectory saved to {file_path}"
                )
        else:
            messagebox.showerror(
                "No Data",
                "No trajectory data to save. Please plot the trajectory first.",
            )

    def clear_predefined_fields(self):
        """Clear all predefined trajectory input fields."""
        for param, entry in self.predefined_entries.items():
            if param == "Type":
                entry.set("")
            else:
                entry.delete(0, tk.END)

    def reset_predefined_defaults(self):
        """Reset all predefined trajectory fields to default values."""
        for param, entry in self.predefined_entries.items():
            if param == "Type":
                entry.set(self.predefined_defaults[param])
            else:
                entry.delete(0, tk.END)
                entry.insert(0, self.predefined_defaults[param])


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryGUI(root)
    root.mainloop()
