import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def irl_continuous_state(state_dim, n_actions, discount, transition_probability,
                          trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories with continuous states.

    state_dim: Dimensionality of the state space. int.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (state_dim, A, state_dim).
    trajectories: 3D array of state/action pairs. States and actions are vectors.
        NumPy array with shape (T, L, 2) where T is the number of trajectories
        and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (state_dim,).
    """

    # Define a simple neural network model for reward function approximation
    class RewardModel(tf.keras.Model):
        def __init__(self, state_dim):
            super(RewardModel, self).__init__()
            self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,))
            self.dense2 = tf.keras.layers.Dense(1, activation='linear')

        def call(self, state):
            x = self.dense1(state)
            return self.dense2(x)

    n_trajectories = trajectories.shape[0]

    # Create the reward model
    reward_model = RewardModel(state_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Lists to store the training loss for plotting
    losses = []

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for trajectory in trajectories:
            states = trajectory[:, 0, :]  # Extract states from trajectory
            with tf.GradientTape() as tape:
                predicted_rewards = reward_model(states)
                true_rewards = np.sum(predicted_rewards)
                loss = tf.reduce_mean(tf.square(true_rewards - predicted_rewards))
            gradients = tape.gradient(loss, reward_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, reward_model.trainable_variables))
            total_loss += loss

        average_loss = total_loss / n_trajectories
        losses.append(average_loss.numpy())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss.numpy()}")

    # Get the learned reward function
    learned_reward = reward_model(np.eye(state_dim, dtype=np.float32))

    # Plot the training process
    plot_training_process(losses)

    return learned_reward.numpy().reshape((state_dim,))


def plot_training_process(losses):
    """
    Plot the training loss over epochs.

    losses: List of training losses. List[float].
    """
    plt.plot(losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# Example usage:
state_dim = 12  # Example state dimension (qpos of 12 joints)
n_actions = 12  # Example number of actions (assuming it's the same as state_dim)
discount = 0.9  # Example discount factor
trajectories = np.random.rand(10, 5, 2, state_dim)  # Example trajectories
epochs = 1000  # Example number of epochs
learning_rate = 0.01  # Example learning rate
transition_probability = np.random.rand(state_dim, n_actions, state_dim)  # Example transition probability

learned_reward = irl_continuous_state(state_dim, n_actions, discount,
                                       transition_probability, trajectories,
                                       epochs, learning_rate)

print("Learned Reward:", learned_reward)
