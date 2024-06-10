import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py

class CricketEnv2D(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, max_timesteps=500, T=None,  r=None):
        utils.EzPickle.__init__(self)
        self.timesteps = 0
        self.max_timesteps=max_timesteps
        self.T = T
        self.r = r
        self.prev_obs = None
        self.model = mujoco_py.load_model_from_path("/home/yuchen/insect_walking_irl/envs/assets/Cricket2D-v1.xml")
        mujoco_env.MujocoEnv.__init__(self, '/home/yuchen/insect_walking_irl/envs/assets/Cricket2D-v1.xml', 2)

    def step(self, action):
        vel = self.sim.data.qvel.flat[0]
        forward_reward = vel
        if (self._get_obs is not None) and (self.r is not None):
            reward_network = self.r(self._get_obs().copy())
        else:
            reward_network = 0
        self.do_simulation(action, self.frame_skip)

        ctrl_cost = .01 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        state = self.state_vector()
        flipped = not (state[2] >= 0.2) 
        flipped_rew = -1 if flipped else 0
        reward = forward_reward - ctrl_cost - contact_cost +flipped_rew

        if self.r is not None:
            # print(reward_network)
            reward = reward_network
        # self.prev_obs = self._get_obs().copy()
        self.timesteps += 1
        done = self.timesteps >= self.max_timesteps

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_flipped=flipped_rew)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[-12:].flat,
            self.sim.data.qvel[-12:].flat,
        ])
    
    def reset_model(self):
        self.timesteps = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_obs = self._get_obs().copy()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths):
        forward_rew = np.array([np.mean(traj['env_infos']['reward_forward']) for traj in paths])
        reward_ctrl = np.array([np.mean(traj['env_infos']['reward_ctrl']) for traj in paths])
        reward_cont = np.array([np.mean(traj['env_infos']['reward_contact']) for traj in paths])
        reward_flip = np.array([np.mean(traj['env_infos']['reward_flipped']) for traj in paths])


if __name__ == "__main__":
    env = CricketEnv2D(max_timesteps=500)

    for _ in range(500):
        env.render()
        env.step(env.action_space.sample())