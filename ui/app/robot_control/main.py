import argparse
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from interface import SimulatedRobot
from robot import Robot
from guesser import GuessingBot

def main(end_effector='joint6'):
    '''initialize'''
    np.set_printoptions(precision=6, suppress=True)
    
    # Define model and data in mujoco
    urdf_path='low_cost_robot/scene.xml'
    device_name='/dev/ttyACM0'

    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)
    
    # Make GueesingBot
    guesser = GuessingBot(
        model=model,
        data=data,
        device_name=device_name,
        end_effector=end_effector
    )

    # Move real robot to home position
    guesser.move_to_home()
    
    '''Start guessing game'''
    print(">>> Game start!!!!")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():   
            # TODO
            # get real target point
            # target point = 
            target_point = [-0.15, 0.15, 0.1]
            guesser.move_to_target(target_point, viewer=viewer)
            guesser.pick_and_place()
            break
    
    print('>>> Finished.')
    guesser.finish()


if __name__ == '__main__':
    main()