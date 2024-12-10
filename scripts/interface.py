from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import mujoco
import numpy as np

PI = 3.14159265

class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.m = m
        self.d = d

        # Define the kinematic chain based on the MuJoCo XML
        self.robot_chain = Chain(name='robot', links=[
            OriginLink(),
            URDFLink(
                name="joint1",
                origin_translation=[0, 0.07, 0],    # Translation vector for joint1
                origin_orientation=[0, 0, 0],       # Orientation (roll, pitch, yaw) for joint1
                rotation=[0, 0, 1],                 # Rotation axis for joint1
                # bounds=(-np.pi/2, np.pi/2)          # Joint angle bounds (example: -90 to 90 degrees in radians)
            ),
            URDFLink(
                name="joint2",
                origin_translation=[0, 0, 0.055],   # Translation vector for joint2
                origin_orientation=[0, 0, 0],       # Orientation for joint2
                rotation=[1, 0, 0],                 # Rotation axis for joint2
                # bounds=(-np.pi/2, np.pi/2)
            ),
            URDFLink(
                name="joint3",
                origin_translation=[0, 0.0135, 0.108],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                # bounds=(-np.pi/2, np.pi/2)
            ),
            URDFLink(
                name="joint4",
                origin_translation=[0, 0.091, 0.0035],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                # bounds=(-np.pi/2, np.pi/2)
            ),
            URDFLink(
                name="joint5",
                origin_translation=[0, 0.04, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                # bounds=(-np.pi/2, np.pi/2)
            ),
            URDFLink(
                name="joint6",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                # bounds=(-np.pi/2, np.pi/2)
            ),
        ])
        self.ee_offset = np.array([0.0, 0.015, -0.05])

    
    def solve_ik(self, ee_target_pos, ee_target_rot=None, adjustment_rate=0.2, jname=None):
        """
        Solve the inverse kinematics for the robot's end effector with fallback rotation.
        :param ee_target_pos: Target position for the end effector.
        :param ee_target_rot: Optional target orientation for the end effector (3x3 rotation matrix).
                            If None, a default orientation or the current orientation is used.
        :param adjustment_rate: Rate at which joint positions are adjusted towards the solution.
        :param jname: Name of the joint to focus on.
        :return: Adjusted joint angles for the robot.
        """
        if ee_target_rot is None:
            # Default orientation: Identity matrix (no rotation)
            ee_target_rot = np.array([0, 0, 1])

        ee_target_pos = ee_target_pos - self.ee_offset

        # Solve IK using IKPy
        ik_solution = self.robot_chain.inverse_kinematics(
            target_position=ee_target_pos
            # , target_orientation= ee_target_rot
        )
        ik_solution = ik_solution[1:7]  # Only active joint positions

        # Apply adjustment rate
        # current_joint_positions = self.d.qpos[:6]  # Current joint positions for active joints
        # adjusted_joint_positions = (
        #     current_joint_positions + adjustment_rate * (ik_solution - current_joint_positions)
        # )

        return ik_solution


    def apply_joint_positions(self, joint_positions):
        """
        Apply joint positions to the MuJoCo model.
        :param joint_positions: Joint angles calculated by the IK solver.
        """
        for i, angle in enumerate(joint_positions[:]):  # Skip the OriginLink (index 0)
            self.d.qpos[i] = angle
        mujoco.mj_step(self.m, self.d)



    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        for i in range(len(pos)):
            if pos[i] < -PI:
                pos[i] += 2 * PI
            elif pos[i] > PI:
                pos[i] -= 2 * PI
        return ((pos + PI) / (2 * PI)) * 4096

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        """
        :param pwm: numpy array of pwm values in range [0, 4096]
        :return: numpy array of joint positions in range [-pi, pi]
        """
        return (pwm / 2048 - 1) * 3.14 

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of pwm values in range [0, 4096]
        :return: numpy array of values in range [0, 1]
        """
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of values in range [0, 1]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return x * 4096

    def read_position(self) -> np.ndarray:
        """ㄴ
        :return: numpy array of current joint positions in range [0, 4096]
        """
        return self.d.qpos[:6] # 5-> 6

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def read_ee_pos(self, joint_name='end_effector'):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        joint_id = self.m.body(joint_name).id
        return self.d.geom_xpos[joint_id]

    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos

    def inverse_kinematics(self, ee_target_pos, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        
        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        
        # compute target joint velocities
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos) # 5->6 due to increased njoints
        
        # apply the joint velocities
        qpos = self.read_position()
        q_target_pos = qpos + qdot * rate
        return q_target_pos

    def inverse_kinematics_rot(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(6)
        integration_dt = 1.0

        # compute the jacobian
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        
        # compute target joint velocities
        jac = np.vstack([jacp, jacr])

        # Orientation error.
        
        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)

        mujoco.mju_negQuat(site_quat_conj, site_quat)

        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)

        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])
        
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        # Set the control signal.
        np.clip(q[:6], *self.m.jnt_range.T[:, :6], out=q[:6])
        self.d.ctrl[:6] = q[:6]

        # Step the simulation.
        mujoco.mj_step(self.m, self.d)

