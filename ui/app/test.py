import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('./ui/app/robot_control/low_cost_robot/scene.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        True