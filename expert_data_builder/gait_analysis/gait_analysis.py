import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# temperary test for c21-0680 data
fold_path = os.getcwd() + '/expert_data_builder'
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join(fold_path, 'joint_movement', cricket_number, f'PIC{video_number}_Joint_movement.csv')
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0])

def plot_gait_phase_diagram(time, joint_angles):
    # Define the threshold angle to distinguish between stance and swing phase
    threshold_angle = 0  # You may need to adjust this based on your data

    # Identify stance and swing phases based on the threshold angle
    stance_phase = joint_angles > threshold_angle
    swing_phase = joint_angles <= threshold_angle

    # Plot the joint angles
    plt.plot(time, joint_angles, label='Joint Angles')

    # Highlight stance and swing phases
    plt.fill_between(time, joint_angles, where=stance_phase, color='green', alpha=0.3, label='Stance Phase')
    plt.fill_between(time, joint_angles, where=swing_phase, color='orange', alpha=0.3, label='Swing Phase')

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Joint Angles')
    plt.title('Gait Phase Diagram')
    plt.legend()

    # Show the plot
    plt.show()

# Example usage
# Replace these with your actual time and joint angle data
time_data = np.linspace(0, 10, 100)
joint_angle_data = np.sin(time_data)  # Replace this with your actual joint angle data

# Call the function to plot the gait phase diagram
plot_gait_phase_diagram(time_data, joint_angle_data)
