import time
import mujoco
import mujoco.viewer
import numpy as np
from interface import SimulatedRobot
from robot import Robot
# from safety import RobotSafety


step = 0
stop_iter = 10000
adjustment_rate = 0.2
prev_error = np.inf
np.set_printoptions(precision=6, suppress=True)

# PID states
Kp = 0.2  # Proportional gain
Ki = 0.01  # Integral gain
Kd = 0.005  # Derivative gain
previous_error = np.zeros(6)
integral_error = np.zeros(6)

# initialize
model = mujoco.MjModel.from_xml_path('robot_control/mujoco/low_cost_robot/scene.xml')
data = mujoco.MjData(model)

# simulator
sim_robot = SimulatedRobot(model, data)
end_effector = 'joint6'  # end-effector joint name
body_id = mujoco.mj_name2id(sim_robot.m, mujoco.mjtObj.mjOBJ_BODY, end_effector)
initial_position = data.geom_xpos[body_id]

# real robot
real_robot = Robot(device_name='/dev/ttyACM0')
real_robot._set_position_control()
real_robot._enable_torque()

# Move sim_robot to current position of real robot
pwm = np.array(real_robot.read_position())
current_qpos = sim_robot._pwm2pos(pwm)
sim_robot.d.qpos[:6] = current_qpos
print(f"Current Joint Position: {current_qpos}")

# Set target point (right: +x, front: +y, up: +z)
# target_point = [0.0, 0.0, 0.4]
# target_point = [0.1, 0.2, 0.1]
# target_point = [-0.1, 0.2, 0.1]
target_point = [-0.15, 0.15, 0.2]

# gripper test. Just testing
real_robot._gripper_on()
time.sleep(1)
# real_robot._gripper_off()


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if step >= stop_iter:
            break

        ### Solve IK
        final_rad = sim_robot.solve_ik(target_point)
        current_rad = sim_robot.d.qpos[:6]
        # print("final pose: ",final_rad)

        # Control with PID
        # command_qpos = current_rad + adjustment_rate * (final_rad - current_rad)  # without PID
        ee = final_rad - current_rad
        integral_error += ee * adjustment_rate
        derivative_error = (ee - previous_error) / adjustment_rate
        control_signal = (Kp * ee + Ki * integral_error + Kd * derivative_error)
        command_qpos = current_rad + control_signal
        mujoco.mj_forward(sim_robot.m, sim_robot.d)
        previous_error = ee
        print("goal pos(rad): ", command_qpos)

        # command to real robot
        target_pwm = sim_robot._pos2pwm(command_qpos)
        pwm_int = [int(pwm) for pwm in target_pwm]
        real_robot.set_goal_pos(pwm_int)   
        print('command pos: ', pwm_int)


        ### Sync to simulator
        current_pwm = np.array(real_robot.read_position())
        print('current pos: ', current_pwm)
        sim_robot.d.qpos[:6] = sim_robot._pwm2pos(current_pwm)
        print('current_rad: ',sim_robot.d.qpos[:6])
        mujoco.mj_step(sim_robot.m, sim_robot.d)
        viewer.sync()

        # Calculate error
        current_pos = sim_robot.d.geom_xpos[body_id]
        # print(current_pos)
        error = np.linalg.norm(target_point - current_pos)
        if abs(error - prev_error) <= 1e-5 and error < 0.1:
            print(f"Converged at step {step} with error: {error}")
            break

        prev_error = error
        step += 1

        time.sleep(0.01)  # can change

print('Finished.')

# gripper test. Just testing
#real_robot._gripper_on()
# time.sleep(1)
# real_robot._gripper_off()
# release the robot
# real_robot._disable_torque()