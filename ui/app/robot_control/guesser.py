import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from app.robot_control.interface import SimulatedRobot
from app.robot_control.robot import Robot
from app.robot_control.constants import HOME


class GuessingBot():
    def __init__(
        self,
        model,
        data,
        device_name='/dev/ttyACM0',
        end_effector='joint6'
    ):
        # Robot in simulator
        self.sim_robot = SimulatedRobot(model, data)
        # Robot in real-world
        self.real_robot = Robot(device_name)
        # End-effector joint name
        self.ee = end_effector

        # For safety
        self.track_buffer = 5
        self.qpos_tracker = [None] * self.track_buffer  # position tracker
        # pwm tracker
        self.pwm_tracker = [None] * self.track_buffer
        # acceleration tracker
        self.qacc_tracker = [None] * self.track_buffer
        self.col_tracker = [0] * self.track_buffer

        self.cnt = 0
        self.STOP = False              # safety stop flag
        self.stop_cnt = 0              # counter for maintaining the stop state
        self.restart_cnt = 0           # counter for restart delay
        self.force_cnt = 0             # counter for detecting rigid body motion
        self.stop_buffer = 5           # buffer size for stop handling
        self.vel_revert_buffer = 300   # buffer size for velocity recovery
        self.col_revert_buffer = 500   # buffer size for collision recovery
        self.restart_buffer = 50       # buffer size for restart delay
        self.force_buffer = 5          # threshold buffer for detecting rigid body motion
        self.gravity_thres = 0.03      # threshold for gravity-based motion
        self.force_thres = 0.0075      # threshold for force-based motion
        self.critical_thres = 0.1      # threshold for critical velocity

    def move_to_home(self, steps=1000):
        pwm = np.array(self.real_robot.read_position())
        self.real_robot._set_position_control()
        self.real_robot._enable_torque()
        time.sleep(1.0)

        print("\t[Control] Move to home pose.")
        smooth_traj = np.linspace(pwm, HOME, steps)
        for pwm in smooth_traj:
            self.real_robot.set_goal_pos([int(p) for p in pwm])  # gripper?????
            # time.sleep(0.00001)
            self.pwm_tracker.append(pwm)
            self.qpos_tracker.append(self.sim_robot._pwm2pos(pwm))
        self.real_robot._gripper_on()

    def move_to_target(self, target_point, viewer=None, steps=10, stop_iter=10000):
        prev_error = np.inf
        step = 0

        while viewer.is_running():
            if step >= stop_iter:
                break

            # Plan trajectory
            col_free_trajs = self.sim_robot.plan_traj(
                ee_target_pos=target_point, ee_target_rot=None, steps=steps)
            # Move real robot to target pos
            for i, qpos in enumerate(col_free_trajs):
                # Move to target
                target_pwm = self.sim_robot._pos2pwm(qpos)
                pwm_int = [int(pwm) for pwm in target_pwm]
                self.real_robot.set_goal_pos(pwm_int)

                current_pwm = np.array(self.real_robot.read_position())
                current_qpos = self.sim_robot._pwm2pos(current_pwm)

                # Sync to simulator
                self.sim_robot.sync_with_robot(current_pwm)
                viewer.sync()

                # Update pwm and position tracker to revert
                self.pwm_tracker.append(current_pwm)
                self.qpos_tracker.append(current_qpos)

                if i % 20 == 0:
                    print(
                        f"\t[Control] (Sync) Current pos: {current_pwm} / Current rad: {current_qpos}")

            # Calculate error
            current_point = self.sim_robot.read_ee_pos(joint_name=self.ee)

            error = np.linalg.norm(target_point - current_point)
            if abs(error - prev_error) <= 1e-4 and error < 0.1:
                print(
                    f"\t[Control] Converged at step {step} with error: {error}")
                break

            prev_error = error
            step += 1

            time.sleep(0.01)

        return

    def pick_and_place(self):

        self.real_robot._gripper_on()
        time.sleep(5)
        self.real_robot._gripper_off()

        # Pick the object

        # Move to the position to place the object

        # Place the object

    def finish(self):
        time.sleep(5)
        self.real_robot._disable_torque()
