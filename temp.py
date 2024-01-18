import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def irl_continuous_state(state_dim, trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories with continuous states.

    state_dim: Dimensionality of the state space. int.
    trajectories: 4D array of state/action pairs. States and actions are vectors.
                            NumPy array with shape (T, L, 2, state_dim) where 
                            T is the number of trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (state_dim,).
    """

    # Define a simple neural network model for reward function approximation
    class RewardModel(tf.keras.Model):
        '''
        This model is designed to approximate the unknown reward function.
        '''
        def __init__(self, state_dim):
            super(RewardModel, self).__init__()
            self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,))
            self.dense2 = tf.keras.layers.Dense(1, activation='linear')

        def call(self, state):
            x = self.dense1(state)
            return self.dense2(x)

    n_trajectories = trajectories.shape[0] # 33
    # Create the reward model
    reward_model = RewardModel(state_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # Lists to store the training loss for plotting
    losses = []

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for trajectory in trajectories: # trajectories: (33, 1270, 2, 12)
            states = trajectory[:, 0, :]  # trajectory[:, 0, :]: (1270, 12)
            with tf.GradientTape() as tape:
                predicted_rewards = reward_model(states)
                # Initializing with zeros for continuous states
                true_rewards = tf.constant(0.0, shape=predicted_rewards.shape)  
                loss = tf.reduce_mean(tf.square(true_rewards - predicted_rewards))
            gradients = tape.gradient(loss, reward_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, reward_model.trainable_variables))
            total_loss += loss
        average_loss = total_loss / n_trajectories
        losses.append(average_loss.numpy())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss.numpy()}")

    # Get the learned reward function
    learned_reward = reward_model(np.eye(state_dim, dtype=np.float32))
    # save the training loss
    np.save('losses.npy', np.array(losses))
    # Plot the training process and zoom in the first 100 epochs
    plot_training_process(losses)
    plot_training_process(losses[:100])

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
trajectories = np.load("expert_demo.npy")  #  (33, 1270, 2, 12)
epochs = 1000  # Example number of epochs
learning_rate = 0.01  # Example learning rate

learned_reward = irl_continuous_state(state_dim, trajectories, epochs, learning_rate)
print("Learned Reward:", learned_reward)
