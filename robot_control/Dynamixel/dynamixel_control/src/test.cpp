// terminal #1: ros2 run dynamixel_control read_write_node
// terminal #2: 
// position control:        ros2 topic pub -1 /joint_states sensor_msgs/msg/JointState "data: [0,0,0,0,0,0,0,0]" (radian단위, 정수 입력도 가능)
// torque enable/disable:   ros2 topic pub -1 /command std_msgs/msg/String "data: 'enable'" 

#include <chrono>
#include <cstdlib>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "moveit/move_group_interface/move_group_interface.h"
#include "std_msgs/msg/string.hpp"

// Control table address for XL330-M077
#define ADDR_XL_TORQUE_ENABLE          64
#define ADDR_XL_GOAL_POSITION          116
#define ADDR_XL_PRESENT_POSITION       132

// Default setting
#define BAUDRATE                       1000000
#define DEVICENAME                     "/dev/ttyUSB0"
#define PROTOCOL_VERSION               2.0
#define PI                              3.1415926

#define DXL_MIN_ID                      0
#define DXL_MAX_ID                      5 // Total 6 dynamixels (0 to 5)

#define XL_MIN_POS                      0
#define XL_MAX_POS                      4095
#define GRIPPER_MIN_POS                 0   // need to measure and change
#define GRIPPER_MAX_POS                 600

class DynamixelController : public rclcpp::Node {
public:
  DynamixelController()
    : Node("dxl") {
    portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    // Open port, set port baudrate
    if (!portHandler->openPort()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open the port!");
      return;
    }
    if (!portHandler->setBaudRate(BAUDRATE)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to change the baudrate!");
      return;
    }

    // Enable Dynamixel Torque (write1ByteTxRx(포트?, id, 토크켜는 위치, 값, 에러값))
    for(int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; dxl_id++){
      dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, 1, &dxl_error);
      if (dxl_comm_result != COMM_SUCCESS) {
        packetHandler->getTxRxResult(dxl_comm_result);
      } else if (dxl_error != 0) {
        packetHandler->getRxPacketError(dxl_error);
      } else {
        RCLCPP_INFO(this->get_logger(), "Dynamixel#%d has been successfully connected", dxl_id);
      }
    }

    // Initialize GroupSyncWrite instance
    groupSyncWrite = new dynamixel::GroupSyncWrite(portHandler, packetHandler, ADDR_XL_GOAL_POSITION, 4);

    // Subscribe to the topic
    subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, std::bind(&DynamixelController::topicCallback, this, std::placeholders::_1));

    // Subscribe to the command topic
    command_subscription_ = this->create_subscription<std_msgs::msg::String>(
      "command", 10, std::bind(&DynamixelController::commandCallback, this, std::placeholders::_1));

    // Start the timer - Printing current position of dynamixel every 1000ms
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&DynamixelController::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Dynamixel Controller has been initialized");
  }

  ~DynamixelController() {
    // Disable Dynamixel Torque
    for(int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; dxl_id++){
      dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, 0, &dxl_error);
      if (dxl_comm_result != COMM_SUCCESS) {
        packetHandler->getTxRxResult(dxl_comm_result);
      } else if (dxl_error != 0) {
        packetHandler->getRxPacketError(dxl_error);
      }
    } 

    // Close port
    portHandler->closePort();
    delete groupSyncWrite;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  dynamixel::PortHandler *portHandler;
  dynamixel::PacketHandler *packetHandler;
  dynamixel::GroupSyncWrite *groupSyncWrite;
  uint8_t dxl_error = 0;
  bool dxl_addparam_result = false;
  int32_t dxl_present_position = 0;
  int dxl_comm_result = COMM_TX_FAIL;
  bool torque_enabled_ = true;
  uint8_t joint_sequence[5] = {1, 2, 0, 3, 4}; // 23145

  void timerCallback() {
    // Read present position
    for(int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; dxl_id++){
      dxl_comm_result = packetHandler->read4ByteTxRx(portHandler, dxl_id, ADDR_XL_PRESENT_POSITION, reinterpret_cast<uint32_t*>(&dxl_present_position), &dxl_error);
      if (dxl_comm_result != COMM_SUCCESS) {
        packetHandler->getTxRxResult(dxl_comm_result);
      } else if (dxl_error != 0) {
        packetHandler->getRxPacketError(dxl_error);
      }
      // Display present position
      RCLCPP_INFO(this->get_logger(), "Present Radian of Dynamixel#%d: %f", dxl_id, Pos2Rad(dxl_id, dxl_present_position));
    }
  }

  void topicCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    /////////////////// need to check later ////////////////////
    // Set goal position parameter array from topic messages
    float raw_goal_position[] = {0.0};
    int* param_goal_position = new int[DXL_MAX_ID - DXL_MIN_ID + 1]{0};
    for(int i = 0; i <= (DXL_MAX_ID - DXL_MIN_ID); i++){
      raw_goal_position[i] = msg->position[joint_sequence[i]];
      if(i != 2) // Invert direction for specific joints if needed
        raw_goal_position[i] *= -1;

      // Change radian value to position with integer value and proceed clipping
      param_goal_position[i] = Clipping(i + DXL_MIN_ID, Rad2Pos(raw_goal_position[i]));
      // RCLCPP_INFO(this->get_logger(), "clipped goal position of #%d: %d", i + DXL_MIN_ID, param_goal_position[i]);
    }

    // Add Dynamixel goal position value to the Syncwrite parameter storage
    for(int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; dxl_id++){
      dxl_addparam_result = groupSyncWrite->addParam(dxl_id, reinterpret_cast<uint8_t*>(&param_goal_position[dxl_id - DXL_MIN_ID]));
      if (dxl_addparam_result != true) {
        RCLCPP_ERROR(this->get_logger(), "Failed to add Dynamixel#%d goal position to the Syncwrite parameter storage", dxl_id);
        return;
      }
    }

    // Syncwrite goal position
    dxl_comm_result = groupSyncWrite->txPacket();
    if (dxl_comm_result != COMM_SUCCESS) packetHandler->getTxRxResult(dxl_comm_result);

    // Clear syncwrite parameter storage
    groupSyncWrite->clearParam();
    delete[] param_goal_position;
  }

  void commandCallback(const std_msgs::msg::String::SharedPtr msg) {
    // Torque enable / disable control
    if (msg->data == "enable") {
      enableTorque();
    } else if (msg->data == "disable") {
      disableTorque();

    // Gripper open / close control
    } else if (msg->data == "open"){
      dxl_comm_result = packetHandler->write4ByteTxRx(
        portHandler, 5, ADDR_XL_GOAL_POSITION, GRIPPER_MIN_POS, &dxl_error);
      if (dxl_comm_result == COMM_SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "Gripper opened");
      } else {
        RCLCPP_INFO(this->get_logger(), "Error occurred while opening");
      }

    } else if (msg->data == "close"){
      dxl_comm_result = packetHandler->write4ByteTxRx(
        portHandler, 5, ADDR_XL_GOAL_POSITION, GRIPPER_MAX_POS, &dxl_error);
      if (dxl_comm_result == COMM_SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "Gripper closed");
      } else {
        RCLCPP_INFO(this->get_logger(), "Error occurred while closing");
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Received unknown command: %s", msg->data.c_str());
    }
  }

  void enableTorque() {
    // Enable Dynamixel Torque
    for(int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; dxl_id++){
      dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, 1, &dxl_error);
      if (dxl_comm_result != COMM_SUCCESS) {
        packetHandler->getTxRxResult(dxl_comm_result);
      } else if (dxl_error != 0) {
        packetHandler->getRxPacketError(dxl_error);
      } else {
        RCLCPP_INFO(this->get_logger(), "Dynamixel#%d torque has been enabled", dxl_id);
      }
    }

    torque_enabled_ = true;
  }

  void disableTorque() {
    // Disable Dynamixel Torque
    for(int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; dxl_id++){
      dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, 0, &dxl_error);
      if (dxl_comm_result != COMM_SUCCESS) {
        packetHandler->getTxRxResult(dxl_comm_result);
      } else if (dxl_error != 0) {
        packetHandler->getRxPacketError(dxl_error);
      } else {
        RCLCPP_INFO(this->get_logger(), "Dynamixel#%d torque has been disabled", dxl_id);
      }
    } 

    torque_enabled_ = false;
  }

  int Clipping(uint8_t dxl_id, int goal_position){
    // 0~4: Dynamixels, 5: Gripper
    if(dxl_id <= 4){
      goal_position = goal_position > XL_MAX_POS ? XL_MAX_POS : goal_position;
      goal_position = goal_position < XL_MIN_POS ? XL_MIN_POS : goal_position;
    }else{
      goal_position = goal_position > GRIPPER_MAX_POS ? GRIPPER_MAX_POS : goal_position;
      goal_position = goal_position < GRIPPER_MIN_POS ? GRIPPER_MIN_POS : goal_position;
    }
    return goal_position;
  }

  float Pos2Rad(uint8_t dxl_id, int position){
    float degree = 0.0;
    if(dxl_id <= 4){
      degree = position * PI / float(XL_MAX_POS);
    }else{
      degree = position * PI / float(GRIPPER_MAX_POS);
    }
    return degree;
  }

  int Rad2Pos(float degree){
    int position = static_cast<int>(degree * (XL_MAX_POS / (2 * PI)));
    return position;
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamixelController>());
  rclcpp::shutdown();
  return 0;
}
