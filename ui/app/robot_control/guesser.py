import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from interface import SimulatedRobot
from robot import Robot
from constants import HOME

class GuessingBot():
    def __init__(self, sim_robot, real_robot, end_effector='joint6'):
        self.ee = end_effector
        self.sim_robot = sim_robot
        self.real_robot = real_robot
        self.body_id = mujoco.mj_name2id(self.sim_robot.m, mujoco.mjtObj.mjOBJ_BODY, end_effector)

        # Setting up for real robot
        self.real_robot._set_position_control()
        self.real_robot._enable_torque()    

        # Move sim_robot to current position of real robot
        pwm = np.array(self.real_robot.read_position())
        current_qpos = self.sim_robot._pwm2pos(pwm)
        self.sim_robot.d.qpos[:6] = current_qpos



    def pick_and_place(self, target_point, viewer=None):
        goal_point = [target_point[0], target_point[1] + 0.02, 0.08]
        # print("goal_point: ", goal_point)

        self.move_to_target(goal_point, v=viewer)

        # self.going_down()
        self.move_to_target([goal_point[0], goal_point[1], 0.02], v=viewer)

        self.real_robot._gripper_on()

        # self.going_up()
        self.move_to_target([goal_point[0], goal_point[1], 0.1], v=viewer)

        # self.move_to_goal()
        self.move_to_target([0.15, 0.15, 0.1], v=viewer)

        self.real_robot._gripper_off()

        self.move_to_home()





    def move_to_home(self, steps=150):
        pwm = np.array(self.real_robot.read_position())
        print("Move to home pose.")
        smooth_traj = np.linspace(pwm, HOME, steps)
        for pwm in smooth_traj:
            self.real_robot.set_goal_pos([int(p) for p in pwm])

        curr_pwm = np.array(self.real_robot.read_position())
        curr_pos = self.sim_robot._pwm2pos(curr_pwm)
        self.sim_robot.d.qpos[:6] = curr_pos
        mujoco.mj_forward(self.sim_robot.m,self.sim_robot.d)



    def move_to_target(self, target_point, v=None, steps=100, stop_iter=10000):
        target_ori = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])  # roll 270deg
        prev_error = np.inf
        step = 0
        stop_flag = False

        # Read initial PWM and solve IK for the target
        curr_pwm = np.array(self.real_robot.read_position())
        final_rad = self.sim_robot.solve_ik(target_point, target_ori)
        final_pwm = self.sim_robot._pos2pwm(final_rad)

        while v.is_running() and not stop_flag:
            if step >= stop_iter:
                print("[Control] Reached maximum iterations. Stopping.")
                break
            traj = list(np.linspace(curr_pwm, final_pwm, steps))  # Generate trajectory

            # Move along the trajectory
            for pwm in traj:
                self.real_robot.set_goal_pos([int(p) for p in pwm])
                curr_pwm = np.array(self.real_robot.read_position())
                curr_pos = self.sim_robot._pwm2pos(curr_pwm)
                self.sim_robot.d.qpos[:6] = curr_pos
                mujoco.mj_forward(self.sim_robot.m, self.sim_robot.d)

                ### Sync to simulator
                mujoco.mj_step(self.sim_robot.m, self.sim_robot.d)
                v.sync()

                # Calculate error
                current_point = self.sim_robot.d.geom_xpos[self.body_id]
                error = np.linalg.norm(target_point - current_point)

                # Check convergence or need for PID adjustment
                if abs(error - prev_error) <= 1e-5 and error < 0.035:
                    print(f"[Control] Converged at step {step} with error: {error}")
                    stop_flag = True
                    break

                prev_error = error
                step += 1
                # print(f"Error: {error}")
                time.sleep(0.005)


            # Adjust using PID Control
            if not stop_flag:
                pid = True
                while pid:
                    cur_pwm = np.array(self.real_robot.read_position())
                    control_signal = self.sim_robot.control_pid(cur_pwm, final_pwm, 
                                                                Kp=0.5, Ki=0.1, Kd=0.01, adjust_rate=0.1)
                    final_pwm = curr_pwm + control_signal
                    pwm_int = [int(pwm) for pwm in final_pwm]
                    self.real_robot.set_goal_pos(pwm_int)

                    self.sim_robot.d.qpos[:6] = self.sim_robot._pwm2pos(cur_pwm)
                    mujoco.mj_step(self.sim_robot.m, self.sim_robot.d)
                    v.sync()

                    # Calculate error
                    current_point = self.sim_robot.d.geom_xpos[self.body_id]
                    error = np.linalg.norm(target_point - current_point)
                    print(f"Error: {error}")
                    if error < 0.035:
                        pid = False
                        stop_flag = True



    def going_down(self, steps=100):
        pwm = np.array(self.real_robot.read_position())
        adj_pwm = pwm[:]
        traj = list(np.linspace(pwm[1], 1250, steps))
        for p in traj:
            adj_pwm[1] = p
            self.real_robot.set_goal_pos([int(p) for p in pwm])
            curr_pwm = np.array(self.real_robot.read_position())
            curr_pos = self.sim_robot._pwm2pos(curr_pwm)
            self.sim_robot.d.qpos[:6] = curr_pos
            mujoco.mj_forward(self.sim_robot.m,self.sim_robot.d)


    def going_up(self, steps=100):
        pwm = np.array(self.real_robot.read_position())
        adj_pwm = pwm[:]
        traj = list(np.linspace(pwm[1], 2048, steps))
        for p in traj:
            adj_pwm[1] = p
            self.real_robot.set_goal_pos([int(p) for p in pwm])
            curr_pwm = np.array(self.real_robot.read_position())
            curr_pos = self.sim_robot._pwm2pos(curr_pwm)
            self.sim_robot.d.qpos[:6] = curr_pos
            mujoco.mj_forward(self.sim_robot.m,self.sim_robot.d)


    def move_to_goal(self, steps=300):
        home_place = [1278, 1689, 2743, 743, 2794, 1123]
        curr_pwm = np.array(self.real_robot.read_position())
        traj = list(np.linspace(curr_pwm, home_place, steps))
        for pwm in traj:
            self.real_robot.set_goal_pos([int(p) for p in pwm])
            curr_pwm = np.array(self.real_robot.read_position())
            curr_pos = self.sim_robot._pwm2pos(curr_pwm)
            self.sim_robot.d.qpos[:6] = curr_pos
            mujoco.mj_forward(self.sim_robot.m,self.sim_robot.d)
