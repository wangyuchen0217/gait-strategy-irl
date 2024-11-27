import numpy as np
import matplotlib.pyplot as plt

q_value_ante_carausius = np.loadtxt('test_folder/maxent/Carausius/ante_gait/q_values_maxent_direction.csv', delimiter=',')
q_value_dirvel_carausius = np.loadtxt('test_folder/maxent/Carausius/dirvel_gait/q_values_maxent_direction.csv', delimiter=',')
q_value_accvel_carausius = np.loadtxt('test_folder/maxent/Carausius/accvel_gait/q_values_maxent_velocity.csv', delimiter=',')
q_value_dirvel_aretaon = np.loadtxt('test_folder/maxent/Aretaon/dirvel_gait/q_values_maxent_direction.csv', delimiter=',')
q_value_accvel_aretaon = np.loadtxt('test_folder/maxent/Aretaon/accvel_gait/q_values_maxent_velocity.csv', delimiter=',')
q_value_dirvel_medauroidea = np.loadtxt('test_folder/maxent/Medauroidea/dirvel_gait/q_values_maxent_direction.csv', delimiter=',')
q_value_accvel_medauroidea = np.loadtxt('test_folder/maxent/Medauroidea/accvel_gait/q_values_maxent_velocity.csv', delimiter=',')

trajectory_ante_carausius = np.loadtxt('test_folder/maxent/Carausius/ante_gait/trajectories.csv', delimiter=',')
trajectory_dirvel_carausius = np.loadtxt('test_folder/maxent/Carausius/dirvel_gait/trajectories.csv', delimiter=',')
trajectory_accvel_carausius = np.loadtxt('test_folder/maxent/Carausius/accvel_gait/trajectories.csv', delimiter=',')
trajectory_dirvel_aretaon = np.loadtxt('test_folder/maxent/Aretaon/dirvel_gait/trajectories.csv', delimiter=',')
trajectory_accvel_aretaon = np.loadtxt('test_folder/maxent/Aretaon/accvel_gait/trajectories.csv', delimiter=',')
trajectory_dirvel_medauroidea = np.loadtxt('test_folder/maxent/Medauroidea/dirvel_gait/trajectories.csv', delimiter=',')
trajectory_accvel_medauroidea = np.loadtxt('test_folder/maxent/Medauroidea/accvel_gait/trajectories.csv', delimiter=',')

state_indices_ante_carausius = trajectory_ante_carausius[:, 0].astype(int)
state_indices_dirvel_carausius = trajectory_dirvel_carausius[:, 0].astype(int)
state_indices_accvel_carausius = trajectory_accvel_carausius[:, 0].astype(int)
actions_carausius = trajectory_ante_carausius[:, 1].astype(int)
state_indices_dirvel_aretaon = trajectory_dirvel_aretaon[:, 0].astype(int)
state_indices_accvel_aretaon = trajectory_accvel_aretaon[:, 0].astype(int)
actions_aretaon = trajectory_dirvel_aretaon[:, 1].astype(int)
state_indices_dirvel_medauroidea = trajectory_dirvel_medauroidea[:, 0].astype(int)
state_indices_accvel_medauroidea = trajectory_accvel_medauroidea[:, 0].astype(int)
actions_medauroidea = trajectory_dirvel_medauroidea[:, 1].astype(int)


# q_values = q_value_ante[state_indices_ante]+q_value_dirvel[state_indices_dirvel]+q_value_accvel[state_indices_accvel]
# q_values = (
#     (q_value_ante[state_indices_ante] * (q_value_ante[state_indices_ante] > 0.5)) +
#     (q_value_dirvel[state_indices_dirvel] * (q_value_dirvel[state_indices_dirvel] > 0.5)) +
#     (q_value_accvel[state_indices_accvel] * (q_value_accvel[state_indices_accvel] > 0.5))
# )
# print(q_values.shape)


# the policy used
q_values = q_value_accvel_medauroidea
# the test source
state_indices = state_indices_accvel_carausius
actions = actions_carausius

replicated_trajectory = []
for i in range(len(state_indices)):
    action_probabilities = q_values[state_indices[i]]
    # Select the action with the highest probability (greedy policy)
    action = np.argmax(action_probabilities)
    replicated_trajectory.append(action)
replicated_trajectory = np.array(replicated_trajectory)


# Plot a heat map to show the trajectory using imshow
plt.figure(figsize=(10, 3))
plt.imshow(q_values[state_indices].T, cmap="plasma", aspect='auto')
plt.title("Heatmap of Action Probabilities along the Expert Trajectory")
plt.xlabel("Trajectory Step Index")
plt.ylabel("Action Index")
plt.gca().invert_yaxis()
plt.colorbar()
plt.tight_layout()
plt.savefig('actions_probability_trajectory.png')

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.eventplot([np.where(replicated_trajectory == i)[0] for i in range(6)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
plt.yticks(range(6), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5"])
plt.xlabel("Trajectory Step Index")
plt.ylabel("Action")
plt.title("Actions along the Replicated Trajectory")
plt.subplot(2, 1, 2)
plt.eventplot([np.where(actions == i)[0] for i in range(6)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
plt.yticks(range(6), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5"])
plt.xlabel("Trajectory Step Index")
plt.ylabel("Action")
plt.title("Actions along the Expert Trajectory")
plt.tight_layout()
plt.savefig('actions_along_trajectories.png')

