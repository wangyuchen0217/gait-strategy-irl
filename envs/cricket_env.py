import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py


def cricket_env(gear=150):
    #mjcmodel = MJCModel('ant_maze')
    mjcmodel = mujoco_py.load_model_from_path("envs/assets/cricket.xml")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
    asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
    asset.texture(builtin="checker",height="100",name="texplane",rgb1="0.25 0.25 0.25",rgb2="0.3 0.3 0.3",type="2d",width="100",mark="cross",markrgb="0.8 0.8 0.8")
    asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
    asset.material(name="geom",texture="texgeom",texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(cutoff="100",diffuse=[1,1,1],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
    worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.8 0.8 1",size="40 40 40",type="plane")

    cricket = worldbody.body(name='torso', pos=[0, 0, 0.75])
    cricket.geom(name='torso_geom', pos=[0, 0, 0], size="0.3 0.2 0.08", type="box")
    cricket.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")
    cricket.camera(name="track", mode="trackcom", pos=[0, -3, 0.3], xyaxes=[1, 0, 0, 0, 0, 1])

    LF_leg = cricket.body(name="LF_leg", pos=[0, 0, 0])
    LF_leg.geom(fromto=[0.0, 0.0, 0.0, 0.3, 0.2, 0.0], name="LF_aux_geom", size="0.08", type="capsule")
    LF_aux = LF_leg.body(name="LF_aux_geom", pos=[0.3, 0.2, 0])
    LF_aux.joint(axis=[1, -1, 0], name="LF_CTr", pos=[0.0, 0.0, 0.0], range=[-10, 10], type="hinge")
    LF_aux.joint(axis=[0, 0, -1], name="LF_hip", pos=[0.0, 0.0, 0.0], range=[-10, 10], type="hinge")
    LF_aux.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="LF_femur_geom", size="0.08", type="capsule")
    LF_knee = LF_aux.body(pos=[0.2, 0.2, 0])
    LF_knee.joint(axis=[-1, 0, 0], name="LF_knee", pos=[0.0, 0.0, 0.0], range=[70, 90], type="hinge")
    LF_knee.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="LF_tibia_geom", size="0.08", type="capsule")

    RF_leg = cricket.body(name="RF_leg", pos=[0, 0, 0])
    RF_leg.geom(fromto=[0.0, 0.0, 0.0, 0.3, -0.2, 0.0], name="RF_aux_geom", size="0.08", type="capsule")
    RF_aux = RF_leg.body(name="RF_aux_geom", pos=[0.3, -0.2, 0])
    RF_aux.joint(axis=[-1, 1, 0], name="RF_CTr", pos=[0.0, 0.0, 0.0], range=[-10, 10], type="hinge")
    RF_aux.joint(axis=[0, 0, 1], name="RF_hip", pos=[0.0, 0.0, 0.0], range=[-10, 10], type="hinge")
    RF_aux.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="RF_femur_geom", size="0.08", type="capsule")
    RF_knee = RF_aux.body(pos=[0.2, -0.2, 0])
    RF_knee.joint(axis=[1, 0, 0], name="RF_knee", pos=[0.0, 0.0, 0.0], range=[70, 90], type="hinge")
    RF_knee.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="RF_tibia_geom", size="0.08", type="capsule")

    LM_leg = cricket.body(name="LM_leg", pos=[0, 0, 0])
    LM_leg.geom(fromto=[0.0, 0.0, 0.0, 0.0, 0.2, 0.0], name="LM_aux_geom", size="0.08", type="capsule")
    LM_aux = LM_leg.body(name="LM_aux_geom", pos=[0.0, 0.2, 0])
    LM_aux.joint(axis=[1, 0, 0], name="LM_CTr", pos=[0.0, 0.0, 0.0], range=[-10, 20], type="hinge")
    LM_aux.joint(axis=[0, 0, 1], name="LM_hip", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    LM_aux.geom(fromto=[0.0, 0.0, 0.0, 0.0, 0.3, 0.0], name="LM_femur_geom", size="0.08", type="capsule")
    LM_knee = LM_aux.body(pos=[0.0, 0.3, 0.0])
    LM_knee.joint(axis=[-1, 0, 0], name="LM_knee", pos=[0.0, 0.0, 0.0], range=[70, 80], type="hinge")
    LM_knee.geom(fromto=[0.0, 0.0, 0.0, 0.0, 0.3, 0.0], name="LM_tibia_geom", size="0.08", type="capsule")

    RM_leg = cricket.body(name="RM_leg", pos=[0, 0, 0])
    RM_leg.geom(fromto=[0.0, 0.0, 0.0, 0.0, -0.2, 0.0], name="RM_aux_geom", size="0.08", type="capsule")
    RM_aux = RM_leg.body(name="RM_aux_geom", pos=[0.0, -0.2, 0])
    RM_aux.joint(axis=[-1, 0, 0], name="RM_CTr", pos=[0.0, 0.0, 0.0], range=[-10, 20], type="hinge")
    RM_aux.joint(axis=[0, 0, 1], name="RM_hip", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
    RM_aux.geom(fromto=[0.0, 0.0, 0.0, 0.0, -0.3, 0.0], name="RM_femur_geom", size="0.08", type="capsule")
    RM_knee = RM_aux.body(pos=[0.0, -0.3, 0.0])
    RM_knee.joint(axis=[1, 0, 0], name="RM_knee", pos=[0.0, 0.0, 0.0], range=[70, 80], type="hinge")
    RM_knee.geom(fromto=[0.0, 0.0, 0.0, 0.0, -0.3, 0.0], name="RM_tibia_geom", size="0.08", type="capsule")

    LH_leg = cricket.body(name="LH_leg", pos=[0, 0, 0])
    LH_leg.geom(fromto=[0.0, 0.0, 0.0, -0.3, 0.2, 0.0], name="LH_aux_geom", size="0.08", type="capsule")
    LH_aux = LH_leg.body(name="LH_aux_geom", pos=[-0.3, 0.2, 0])
    LH_aux.joint(axis=[3, 5, 0], name="LH_CTr", pos=[0.0, 0.0, 0.0], range=[-10, 0], type="hinge")
    LH_aux.joint(axis=[0, 0, 1], name="LH_hip", pos=[0.0, 0.0, 0.0], range=[0, 10], type="hinge")
    LH_aux.geom(fromto=[0.0, 0.0, 0.0, -0.45, 0.25, 0.3], name="LH_femur_geom", size="0.08", type="capsule")
    LH_knee = LH_aux.body(pos=[-0.45, 0.25, 0.3])
    LH_knee.joint(axis=[-1, 0, 0], name="LH_knee", pos=[0.0, 0.0, 0.0], range=[70, 150], type="hinge")
    LH_knee.geom(fromto=[0.0, 0.0, 0.0, -0.45, 0.25, 0.0], name="LH_tibia_geom", size="0.08", type="capsule")

    RH_leg = cricket.body(name="RH_leg", pos=[0, 0, 0])
    RH_leg.geom(fromto=[0.0, 0.0, 0.0, -0.3, -0.2, 0.0], name="RH_aux_geom", size="0.08", type="capsule")
    RH_aux = RH_leg.body(name="RH_aux_geom", pos=[-0.3, -0.2, 0])
    RH_aux.joint(axis=[-3, 5, 0], name="RH_CTr", pos=[0.0, 0.0, 0.0], range=[-10, 0], type="hinge")
    RH_aux.joint(axis=[0, 0, -1], name="RH_hip", pos=[0.0, 0.0, 0.0], range=[0, 10], type="hinge")
    RH_aux.geom(fromto=[0.0, 0.0, 0.0, -0.45, -0.25, 0.3], name="RH_femur_geom", size="0.08", type="capsule")
    RH_knee = RH_aux.body(pos=[-0.45, -0.25, 0.3])
    RH_knee.joint(axis=[1, 0, 0], name="RH_knee", pos=[0.0, 0.0, 0.0], range=[70, 150], type="hinge")
    RH_knee.geom(fromto=[0.0, 0.0, 0.0, -0.45, -0.25, 0.0], name="RH_tibia_geom", size="0.08", type="capsule")

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LF_CTr", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LF_hip", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LF_knee", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RF_CTr", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RF_hip", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RF_knee", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LM_CTr", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LM_hip", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LM_knee", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RM_CTr", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RM_hip", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RM_knee", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LH_CTr", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LH_hip", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="LH_knee", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RH_CTr", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RH_hip", gear=gear)
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="RH_knee", gear=gear)
    return mjcmodel


class CustomAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    A modified ant env with lower joint gear ratios so it flips less often and learns faster.
    """
    def __init__(self, max_timesteps=500, r=None, disabled=False, gear=150, T=None):
        #mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        self.timesteps = 0
        self.max_timesteps=max_timesteps
        self.T = T
        self.r = r
        self.prev_obs = None
        model = cricket_env(gear=gear)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        vel = self.sim.data.qvel.flat[0]
        forward_reward = vel
        if (self._get_obs is not None) and (self.r is not None):
            reward_network = self.r(self._get_obs().copy())
        else:
            reward_network = 0
        self.do_simulation(a, self.frame_skip)

        ctrl_cost = .01 * np.square(a).sum()
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
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
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
    env = CustomAntEnv(disabled=False, gear=30)

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
