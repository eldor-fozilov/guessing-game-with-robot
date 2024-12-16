import argparse
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from interface import SimulatedRobot
from robot import Robot
from guesser import GuessingBot
from img2world import CalibBoard

def main(end_effector='joint6'):
    '''initialize'''
    np.set_printoptions(precision=6, suppress=True)
    
    # Define model and data in mujoco
    urdf_path='low_cost_robot/scene.xml'
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)

    # Setting Robot
    sim_robot = SimulatedRobot(model, data)
    real_robot = Robot(device_name='/dev/ttyACM0')
    guesser = GuessingBot(sim_robot, real_robot, end_effector=end_effector)
    t = CalibBoard()

    # Move real robot to home position
    guesser.move_to_home()
    real_robot._gripper_off()

    '''Start guessing game'''
    print("Game start!!!!")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():   
            
            
            image_pixel = [320,240]
            target_point = t.cam2robot(image_pixel[0], image_pixel[1])
            guesser.pick_and_place(target_point, viewer=viewer)
            
            
            return
    print('Finished.')
    real_robot._disable_torque()


if __name__ == '__main__':
    main()