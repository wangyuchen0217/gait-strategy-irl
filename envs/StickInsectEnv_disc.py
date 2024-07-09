import numpy as np
import torch
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Discrete
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class StickInsectEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(
        self,
        xml_file="/home/yuchen/insect_walking_irl/envs/assets/StickInsect-v0.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 3.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        state_dim=10,
        action_dim=48,
        n_bins=2,
        pca_dimension=10,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_space = Discrete(n_bins ** state_dim)
        self.action_space = Discrete(action_dim)
        self.observation_matrix = np.eye(n_bins ** state_dim)
        self.transition_matrix = np.zeros((n_bins ** state_dim, action_dim, n_bins ** state_dim))
        self.initial_state_dist = np.zeros(n_bins ** state_dim)
        self.pca_dimension = pca_dimension
        self.n_bins = n_bins

        MujocoEnv.__init__(
            self,
            xml_file,
            1,
            observation_space=self.state_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )


    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    # def step(self, action):
    #     xy_position_before = self.get_body_com("torso")[:2].copy()
    #     self.do_simulation(action, self.frame_skip)
    #     xy_position_after = self.get_body_com("torso")[:2].copy()

    #     xy_velocity = (xy_position_after - xy_position_before) / self.dt
    #     x_velocity, y_velocity = xy_velocity

    #     forward_reward = x_velocity * 10
    #     healthy_reward = self.healthy_reward

    #     rewards = forward_reward + healthy_reward

    #     costs = ctrl_cost = self.control_cost(action)

    #     terminated = self.terminated
    #     observation = self._get_obs()
    #     info = {
    #         "reward_forward": forward_reward,
    #         "reward_ctrl": -ctrl_cost,
    #         "reward_survive": healthy_reward,
    #         "x_position": xy_position_after[0],
    #         "y_position": xy_position_after[1],
    #         "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
    #         "x_velocity": x_velocity,
    #         "y_velocity": y_velocity,
    #         "forward_reward": forward_reward,
    #     }
    #     if self._use_contact_forces:
    #         contact_cost = self.contact_cost
    #         costs += contact_cost
    #         info["reward_ctrl"] = -contact_cost

    #     reward = rewards - costs

    #     if self.render_mode == "human":
    #         self.render()
    #     # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
    #     return observation, rewards, terminated, False, info

    # def _get_obs(self):
    #     position = self.data.qpos.flat.copy()
    #     velocity = self.data.qvel.flat.copy()

    #     if self._exclude_current_positions_from_observation:
    #         position = position[2:]

    #     if self._use_contact_forces:
    #         contact_force = self.contact_forces.flat.copy()
    #         return np.concatenate((position, velocity, contact_force))
    #     else:
    #         return np.concatenate((position, velocity))

    # def reset_model(self):
    #     # noise_low = -self._reset_noise_scale
    #     # noise_high = self._reset_noise_scale

    #     # qpos = self.init_qpos + self.np_random.uniform(
    #     #     low=noise_low, high=noise_high, size=self.model.nq
    #     # )
    #     # qvel = (
    #     #     self.init_qvel
    #     #     + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
    #     # )
    #     # self.set_state(qpos, qvel)

    #     # observation = self._get_obs()

    #     qpos = self.init_qpos
    #     qvel = self.init_qvel
    #     self.set_state(qpos, qvel)
    #     observation = self._get_obs()

    #     return observation

    def reset(self):
        obs = self.original_env.reset()
        return self._transform_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.original_env.step(action)
        transformed_obs = self._transform_observation(obs)
        return transformed_obs, reward, done, info

    def _transform_observation(self, obs):
        if self.exclude_xy:
            obs = obs[2:]  # Exclude the first two elements if exclude_xy is True
        scaled_obs = self.scaler.transform([obs])  # Scale the observation
        pca_obs = self.pca.transform(scaled_obs)  # Apply PCA
        return pca_obs.flatten()

if __name__ == "__main__":
    env = StickInsectEnv(render_mode='human')
    env.reset_model()

    # print the observation space and action space
    print("observation space:", env.observation_space)
    print("observation space shape:", env.observation_space.shape)
    print("action space:", env.action_space)
    print("action space shape:", env.action_space.shape)

    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()
    env.close()