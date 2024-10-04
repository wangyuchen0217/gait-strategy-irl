import numpy as np
import pandas as pd


def dynamic_phase_diff(leg1_stance_times, leg2_stance_times):
    phase_diffs = []
    
    # Loop through stance times of leg 1
    for i in range(len(leg1_stance_times) - 1):
        # Leg 1 cycle duration (current stance to next stance)
        leg1_cycle_duration = leg1_stance_times[i+1] - leg1_stance_times[i]
        
        # Find the first stance of leg 2 that occurs during leg 1's cycle
        for j in range(len(leg2_stance_times)):
            if leg2_stance_times[j] >= leg1_stance_times[i] and leg2_stance_times[j] < leg1_stance_times[i+1]:
                # Time difference between leg 1 and leg 2 stance starts
                time_diff = leg2_stance_times[j] - leg1_stance_times[i]
                
                # Calculate phase difference (relative to leg 1's cycle duration)
                phase_diff = (time_diff % leg1_cycle_duration) / leg1_cycle_duration
                
                # Convert to radians
                phase_diff_radians = phase_diff * 2 * np.pi
                phase_diffs.append(phase_diff_radians)
                break
    
    return np.array(phase_diffs)

# Function to detect transitions from swing (0) to stance (1) for each leg
def detect_transitions(leg_data):
    transitions = []
    for i in range(1, len(leg_data)):
        if leg_data[i-1] == 0 and leg_data[i] == 1:
            transitions.append(i)  # Record the index of stance phase start
    return transitions



file_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_32_gait.csv'
gait_data = pd.read_csv(file_path)

# Detect transitions for each leg
transitions = {}
legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
for leg in legs:
    transitions[leg] = detect_transitions(gait_data[leg])
print("Transitions detected for each leg:")
for leg, stance_times in transitions.items():
    print(f"{leg}: {stance_times}")
print("---------------------------------")

# Re-run the phase difference calculation now that numpy is imported
phase_diff_LF_RF = dynamic_phase_diff(transitions['LF'], transitions['RF'])
phase_diff_LM_RM = dynamic_phase_diff(transitions['LF'], transitions['LM'])
phase_diff_LH_RH = dynamic_phase_diff(transitions['LF'], transitions['LH'])
print("Phase differences between LF and RF (radians):", phase_diff_LF_RF)
print("Phase differences between LM and RM (radians):", phase_diff_LM_RM)
print("Phase differences between LH and RH (radians):", phase_diff_LH_RH)
print("---------------------------------")
phase_diff_LF_LM = dynamic_phase_diff(transitions['LF'], transitions['LM'])
phase_diff_LF_LH = dynamic_phase_diff(transitions['LF'], transitions['LH'])
phase_diff_RF_RM = dynamic_phase_diff(transitions['RF'], transitions['RM'])
phase_diff_RF_RH = dynamic_phase_diff(transitions['RF'], transitions['RH'])
print("Phase differences between LF and LM (radians):", phase_diff_LF_LM)
print("Phase differences between LF and LH (radians):", phase_diff_LF_LH)
print("Phase differences between RF and RM (radians):", phase_diff_RF_RM)
print("Phase differences between RF and RH (radians):", phase_diff_RF_RH)
print("---------------------------------")

# Bin the Phase Differences: Low/Mid/High Phase Differences
n_bins = 3
phase_diff_bins = np.linspace(0, 2*np.pi, n_bins+1)
phase_diff_LF_RF_binned = np.digitize(phase_diff_LF_RF, phase_diff_bins)
phase_diff_LM_RM_binned = np.digitize(phase_diff_LM_RM, phase_diff_bins)
phase_diff_LH_RH_binned = np.digitize(phase_diff_LH_RH, phase_diff_bins)
print("Binned Phase Differences between LF and RF:", phase_diff_LF_RF_binned)
print("Binned Phase Differences between LM and RM:", phase_diff_LM_RM_binned)
print("Binned Phase Differences between LH and RH:", phase_diff_LH_RH_binned)
print("---------------------------------")
phase_diff_LF_LM_binned = np.digitize(phase_diff_LF_LM, phase_diff_bins)
phase_diff_LF_LH_binned = np.digitize(phase_diff_LF_LH, phase_diff_bins)
phase_diff_RF_RM_binned = np.digitize(phase_diff_RF_RM, phase_diff_bins)
phase_diff_RF_RH_binned = np.digitize(phase_diff_RF_RH, phase_diff_bins)
print("Binned Phase Differences between LF and LM:", phase_diff_LF_LM_binned)
print("Binned Phase Differences between LF and LH:", phase_diff_LF_LH_binned)
print("Binned Phase Differences between RF and RM:", phase_diff_RF_RM_binned)
print("Binned Phase Differences between RF and RH:", phase_diff_RF_RH_binned)
