import time
import numpy as np
import mujoco
import mujoco.viewer

from scripts.interface import SimulatedRobot
from scripts.robot import Robot
import copy

def clock(step_start):
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
    return time.time()

m = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
d = mujoco.MjData(m)
r = SimulatedRobot(m, d)
init_qpos = copy.deepcopy(d.qpos[:6])

robot = Robot(device_name='/dev/ttyACM0')
robot._set_position_control()
robot._enable_torque()

pwm = robot.read_position()
pwm = np.array(pwm)
d.qpos[:6] = r._pwm2pos(pwm)

neutral_pwm = [2048, 2048, 2048, 1024, 2048, 2048]

# initial contact: (0, 11)[floor, redbox], (0, 12)[floor, bluebox]. (1, 2)[base_link, joint1]

track_buffer = 5
qpos_tracker = [r._pwm2pos(pwm)]*track_buffer
pwm_tracker = [pwm]*track_buffer
qacc_tracker = [None]*track_buffer
col_tracker = [0]*track_buffer

stop_iter = 100000

cnt = 0
STOP = False
stop_cnt = 0
restart_cnt = 0
force_cnt = 0
stop_buffer = 5
vel_revert_buffer = 300
col_revert_buffer = 500
restart_buffer = 50
force_buffer = 5
gravity_thres = 0.03
force_thres = 0.0075
critical_thres = 0.1

RUN = True

if RUN:
  print('Please Wait....')
  with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    time.sleep(1.0)
    robot._disable_torque()

    print("Control Start")
    while viewer.is_running():
      cnt += 1

      step_start = time.time()

      # ADD SYNC HERE
      pwm = robot.read_position()
      pwm = np.array(pwm)
      d.qpos[:6] = r._pwm2pos(pwm)

      # save qpos tracker to revert.
      pwm_tracker.append(pwm)
      qpos_tracker.append(r._pwm2pos(pwm))

      # moving to simulator
      mujoco.mj_step(m, d)

      # read velocity of the real robot
      real_vel = robot.read_velocity()
      moving_idx = np.where(np.abs(real_vel)>3)[0]

      # calculate the xyz vel, acc
      if len(moving_idx) > 0:
        qpos = np.array(qpos_tracker)[-track_buffer:]
        qvel = qpos[1:] - qpos[:-1]

        max_vel = np.max(np.abs(qvel))
        if max_vel > force_thres:
          if max_vel > gravity_thres:
            force_cnt += 1

          if force_cnt > force_buffer or np.max(np.abs(qvel)) > critical_thres or STOP:
            if np.max(np.abs(qvel))>critical_thres:
              force_cnt = force_buffer+1 

            if not STOP:
              print('TOO FAST MOVEMENT!', np.max(np.abs(qvel)))
              STOP = True
              
              print('WAIT TO BE STABLIZED')

              # Revert to previous position
              robot._set_position_control()
              robot._enable_torque()
              qpos0 = np.array(robot.read_position())
              if len(pwm_tracker) < vel_revert_buffer:
                qpos1 = neutral_pwm
              else:
                qpos1 = pwm_tracker[-vel_revert_buffer]
              smooth_mover = np.linspace(qpos0, qpos1, vel_revert_buffer)
              for revert_pos in smooth_mover:
                robot.set_goal_pos([int(p) for p in revert_pos])
                step_start = clock(step_start)

                pwm_tracker.append(revert_pos)
                qpos_tracker.append(r._pwm2pos(revert_pos))

              time.sleep(1)
              print('FINISHED TO STABLIZED')

              # SYNC TO SIMULATOR
              print('Synced to Simulator')
              pwm = robot.read_position()
              pwm = np.array(pwm)
              d.qpos[:6] = r._pwm2pos(pwm)
              mujoco.mj_step(m, d)

            else:
              print('[{}/{}] Counting Disturbance...MOVE MORE TO RELEASE TORQUE'.format(restart_cnt, restart_buffer))
              restart_cnt += 1
            
              if restart_cnt >= restart_buffer:
                restart_cnt = 0
                force_cnt = 0
                STOP = False
                robot._set_position_control()
                robot._disable_torque()
                print('RESTART!!')
          
        else:
          pass

      # FIND ADDITIONAL COLLISION
      for g in d.contact.geom:
        if g.tolist() not in [[1, 2]]:
          col_source = d.geom(g[0]).name
          col_target = d.geom(g[1]).name
          print("{} and {} are in collision".format(col_source, col_target))
          print('PLEASE RELEASE YOUR HAND FROM THE ROBOT')

          robot._set_position_control()
          robot._enable_torque()
          time.sleep(2.0)

          col_tracker.append(1)

          # start reverting
          pwm0 = robot.read_position()
          pwm0 = np.array(pwm0)

          if len(pwm_tracker) < col_revert_buffer:
            pwm1 = neutral_pwm
          else:
            pwm1 = pwm_tracker[-col_revert_buffer+1]

          smooth_traj = np.linspace(pwm0, pwm1, col_revert_buffer*5)
          
          # move to get over collision
          print('GETTING OVER COLLISION')
          for pwm in smooth_traj:
            robot.set_goal_pos([int(p) for p in pwm])
            step_start = clock(step_start)
            pwm_tracker.append(pwm)
            qpos_tracker.append(r._pwm2pos(pwm))

          print('RECOVERED FROM COLLISION')

          # SYNC TO SIMULATOR
          print('Synced to Simulator')
          pwm = robot.read_position()
          pwm = np.array(pwm)
          d.qpos[:6] = r._pwm2pos(pwm)
          mujoco.mj_step(m, d)

    
      viewer.sync()

      # Rudimentary time keeping, will drift relative to wall clock.
      step_start = clock(step_start)

      if cnt >= stop_iter:
        break
      
