from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.robots.pendula.inverted_double_pendulum import InvertedDoublePendulum
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene


class InvertedDoublePendulumBulletSparseEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = InvertedDoublePendulum()
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        done = self.robot.pos_y + 0.3 <= 1
        if self.robot.pos_y >= 0.89:
            reward = 1.
        else:
            reward = 0.
        self.HUD(state, a, done)
        return state, reward, done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0,1.2,1.2, 0,0,0.5)
