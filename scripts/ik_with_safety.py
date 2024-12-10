import time
import sys
import copy
import mujoco
import mujoco.viewer
import numpy as np
from interface2 import SimulatedRobot
from scripts.robot import Robot
from safety import RobotSafety

def clock(model, step_start):
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
    return time.time()

def recover_robot():
    real_robot._set_position_control()
    real_robot._enable_torque()
    
    qpos0 = np.array(real_robot.read_position())
    qpos1 = pwm_tracker[-vel_revert_buffer] if len(pwm_tracker) >= vel_revert_buffer else neutral_pwm
    smooth_mover = np.linspace(qpos0, qpos1, vel_revert_buffer)

    for revert_pos in smooth_mover:
        real_robot.set_goal_pos([int(p) for p in revert_pos])
        sim_robot.d.qpos[:6] = sim_robot._pwm2pos(revert_pos)
        mujoco.mj_step(sim_robot.m, sim_robot.d)
        time.sleep(0.01)

    print("Robot stabilized.")

### initialize
model = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
data = mujoco.MjData(model)
init_qpos = copy.deepcopy(data.qpos[:6])

# simulator
sim_robot = SimulatedRobot(model, data)
end_effector = 'joint6'  # end-effector joint name
body_id = mujoco.mj_name2id(sim_robot.m, mujoco.mjtObj.mjOBJ_BODY, end_effector)

# real robot
real_robot = Robot(device_name='/dev/ttyACM0')
real_robot._set_position_control()
real_robot._enable_torque()

# Move sim_robot to current position of real robot
pwm = np.array(real_robot.read_position())
current_qpos = sim_robot._pwm2pos(pwm)
sim_robot.d.qpos[:6] = current_qpos
print(f"Current Joint Position: {current_qpos}")

stop_iter = 10000
step = 0
adjustment_rate = 0.2
prev_error = np.inf

target_point = [0.1, 0.15, 0.15]
target_ori = np.identity(3)

# for safety
neutral_pwm = [2048, 2048, 2048, 1024, 2048, 2048]

track_buffer = 5
qpos_tracker = [sim_robot._pwm2pos(pwm)] * track_buffer  # position tracker
pwm_tracker = [pwm] * track_buffer                       # pwm tracker
qacc_tracker = [None] * track_buffer                     # acceleration tracker
col_tracker = [0] * track_buffer                         # collision state tracker

cnt = 0                     
STOP = False              # safety stop flag
stop_cnt = 0              # counter for maintaining the stop state
restart_cnt = 0           # counter for restart delay
force_cnt = 0             # counter for detecting rigid body motion
stop_buffer = 5           # buffer size for stop handling
vel_revert_buffer = 300   # buffer size for velocity recovery
col_revert_buffer = 500   # buffer size for collision recovery
restart_buffer = 50       # buffer size for restart delay
force_buffer = 5          # threshold buffer for detecting rigid body motion
gravity_thres = 0.03      # threshold for gravity-based motion
force_thres = 0.0075      # threshold for force-based motion
critical_thres = 0.1      # threshold for critical velocity

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    time.sleep(1.0)
    real_robot._disable_torque()

    print("Control Start.")
    while viewer.is_running():
        if step >= stop_iter:
            break
        # ### Add sync to current real robot position
        # current_pwm = np.array(real_robot.read_position())
        # current_qpos = sim_robot._pwm2pos(current_pwm)
        # sim_robot.d.qpos[:6] = current_qpos

        # # Update pwm and position tracker to revert
        # pwm_tracker.append(current_pwm)
        # qpos_tracker.append(sim_robot._pwm2pos(current_pwm))

        # # Moving to simulator
        # mujoco.mj_step(sim_robot.m, sim_robot.d)

        ### Solve IK
        target_qpos = sim_robot.solve_ik(target_point)
        sim_robot.d.qpos[:6] += adjustment_rate * (target_qpos[:6] - sim_robot.d.qpos[:6])
        print(f"[Step {step}] Target Joint Position: {target_qpos}")
        mujoco.mj_forward(sim_robot.m, sim_robot.d)

        # Calculate error
        current_pos = sim_robot.d.geom_xpos[body_id]
        error = np.linalg.norm(target_point - current_pos)
        
        ### FIND ADDITIONAL COLLISION
        # for g in sim_robot.d.contact.geom:
        #     if g.tolist() not in [[1, 2]]:
        #         col_source = sim_robot.d.geom(g[0]).name
        #         col_target = sim_robot.d.geom(g[1]).name
        #         print("{} and {} are in collision".format(col_source, col_target))
        #         print('PLEASE RELEASE YOUR HAND FROM THE ROBOT')

        #         # real_robot._set_position_control()
        #         # real_robot._enable_torque()
        #         # time.sleep(2.0)

        #         col_tracker.append(1)

        #         # start reverting
        #         pwm0 = real_robot.read_position()
        #         pwm0 = np.array(pwm0)

        #         if len(pwm_tracker) < col_revert_buffer:
        #             pwm1 = neutral_pwm
        #         else:
        #             pwm1 = pwm_tracker[-col_revert_buffer+1]

        #         smooth_traj = np.linspace(pwm0, pwm1, col_revert_buffer*5)
                
        #         # move to get over collision
        #         print('GETTING OVER COLLISION')
        #         for pwm in smooth_traj:
        #             real_robot.set_goal_pos([int(p) for p in pwm])
        #             step_start = clock(step_start)
        #             pwm_tracker.append(pwm)
        #             qpos_tracker.append(r._pwm2pos(pwm))

        #         print('RECOVERED FROM COLLISION')

        ### Move real robot to target position
        print("Move to target position.")
        real_robot._set_position_control()
        real_robot._enable_torque()
        
        # sim_robot.d.qpos[:6] += adjustment_rate * (target_qpos[:6] - sim_robot.d.qpos[:6])
        # mujoco.mj_forward(sim_robot.m, sim_robot.d)  

        target_pwm = sim_robot._pos2pwm(sim_robot.d.qpos[:6])
        target_pwm = np.clip(target_pwm, 0, 4096)
        real_robot.set_goal_pos([int(pwm) for pwm in target_pwm])      
        
        ### Sync to simulator
        # print("Synced to Simulator.")
        # current_pwm = np.array(real_robot.read_position())
        # sim_robot.d.qpos[:6] = sim_robot._pwm2pos(current_pwm)
        mujoco.mj_step(sim_robot.m, sim_robot.d)

        viewer.sync()
        time.sleep(1)

        ### Check for convergence
        if abs(error - prev_error) <= 1e-5:
            print(f"Converged at step {step} with error: {error}")
            break

        prev_error = error
        
        step += 1

print('Finished.')