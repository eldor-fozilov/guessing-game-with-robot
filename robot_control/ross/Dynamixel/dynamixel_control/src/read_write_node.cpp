#include <chrono>
#include <cstdlib>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

// Control table address for XL330-M077 and XL430-W250-T
#define ADDR_XL_TORQUE_ENABLE          64
#define ADDR_XL_GOAL_POSITION          116
#define ADDR_XL_PRESENT_POSITION       132

// Default settings
#define BAUDRATE                       1000000
#define DEVICENAME                     "/dev/ttyACM0"
#define PROTOCOL_VERSION               2.0

#define DXL_MIN_ID                      1
#define DXL_MAX_ID                      6

#define POSITION_TO_RAD_CONVERSION      (2 * 3.1415926 / 4096)  // 1 unit = 2π/4096 radians
#define RAD_TO_POSITION_CONVERSION      (4096 / (2 * 3.1415926))  // Radians to Dynamixel position units
#define CENTER_POSITION                 2048  // 2048 is the center position representing 0 degrees

class DynamixelController : public rclcpp::Node {
public:
  DynamixelController()
    : Node("dxl") {
    // Initialize port and packet handlers
    portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    // Open port and set baudrate
    if (!portHandler->openPort()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open port!");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Port opened successfully.");

    if (!portHandler->setBaudRate(BAUDRATE)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set baudrate!");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Baudrate set successfully.");

    // Enable Torque
    for (int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; ++dxl_id) {
      enableTorque(dxl_id);
    }

    // Initialize GroupSyncWrite for goal position
    groupSyncWrite = new dynamixel::GroupSyncWrite(portHandler, packetHandler, ADDR_XL_GOAL_POSITION, 4);

    // Subscribe to topics
    subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, std::bind(&DynamixelController::topicCallback, this, std::placeholders::_1));
    command_subscription_ = this->create_subscription<std_msgs::msg::String>(
      "/command", 10, std::bind(&DynamixelController::commandCallback, this, std::placeholders::_1));

    // Timer to periodically read positions
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&DynamixelController::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Dynamixel Controller initialized.");
  }

  ~DynamixelController() {
    // Disable Torque
    for (int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; ++dxl_id) {
      disableTorque(dxl_id);
    }
    portHandler->closePort();
    delete groupSyncWrite;
    RCLCPP_INFO(this->get_logger(), "Dynamixel Controller terminated.");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  dynamixel::PortHandler *portHandler;
  dynamixel::PacketHandler *packetHandler;
  dynamixel::GroupSyncWrite *groupSyncWrite;
  uint8_t joint_sequence[5] = {1, 2, 3, 4, 5};
  // uint8_t joint_sequence[5] = {2, 3, 1, 4, 5};   // for IK solver

  void enableTorque(int dxl_id) {
    auto result = packetHandler->write1ByteTxRx(portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, 1, nullptr);
    if (result == COMM_SUCCESS) {
      RCLCPP_INFO(this->get_logger(), "Torque enabled for ID %d", dxl_id);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to enable torque for ID %d", dxl_id);
    }
  }
  void disableTorque(int dxl_id) {
    auto result = packetHandler->write1ByteTxRx(portHandler, dxl_id, ADDR_XL_TORQUE_ENABLE, 0, nullptr);
    if (result == COMM_SUCCESS) {
      RCLCPP_INFO(this->get_logger(), "Torque disabled for ID %d", dxl_id);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to disable torque for ID %d", dxl_id);
    }
  }


  void timerCallback() {
    RCLCPP_INFO(this->get_logger(), "---");
    for (int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; ++dxl_id) {
      int32_t position = 0;
      auto result = packetHandler->read4ByteTxRx(portHandler, dxl_id, ADDR_XL_PRESENT_POSITION, reinterpret_cast<uint32_t*>(&position), nullptr);
      if (result == COMM_SUCCESS) {
        float position_in_rad = (position - CENTER_POSITION) * POSITION_TO_RAD_CONVERSION;
        float position_in_deg = position_in_rad * (180.0 / 3.1415926);
        RCLCPP_INFO(this->get_logger(), "DXL#%d Pos: %d (R:%.4f, D:%.2f)", dxl_id, position, position_in_rad, position_in_deg);
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to read position for ID %d", dxl_id);
      }
    }
  }

  void topicCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    for (int i = 0; i < DXL_MAX_ID; ++i) {
      float target_position = msg->position[joint_sequence[i-1]];
      
      // 각도를 Dynamixel 값으로 변환
      int goal_position = static_cast<int>(target_position * RAD_TO_POSITION_CONVERSION) + CENTER_POSITION;
      
      switch (i + DXL_MIN_ID) {
            case 1:
                goal_position = clip_position(goal_position, CENTER_POSITION - 1024, CENTER_POSITION + 1024);
                break;
            case 2:
                goal_position = clip_position(goal_position, CENTER_POSITION - 341, CENTER_POSITION + 1024);
                break;
            case 3:
                goal_position = clip_position(goal_position, CENTER_POSITION - 512, CENTER_POSITION + 1024);
                break;
            case 4:
                goal_position = clip_position(goal_position, CENTER_POSITION, CENTER_POSITION + 2048);
                break;
            case 5:
                goal_position = clip_position(goal_position, CENTER_POSITION - 2048, CENTER_POSITION + 2048);
                break;
            case 6:
                goal_position = clip_position(goal_position, CENTER_POSITION, CENTER_POSITION + 910);
                break;
        }

      // RCLCPP_INFO(this->get_logger(), "will print: %d", goal_position);
      // 목표 위치 설정
      groupSyncWrite->addParam(i + DXL_MIN_ID, reinterpret_cast<uint8_t*>(&goal_position));
    }

    // 전송
    if (groupSyncWrite->txPacket() == COMM_SUCCESS) {
      // RCLCPP_INFO(this->get_logger(), "Goal positions transmitted.");
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to transmit goal positions.");
    }
    groupSyncWrite->clearParam();
  }

  int clip_position(int position, int min_pos, int max_pos) {
    return std::max(min_pos, std::min(max_pos, position));
  }

  void commandCallback(const std_msgs::msg::String::SharedPtr msg) {
    if (msg->data == "enable") {
      for (int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; ++dxl_id) enableTorque(dxl_id);
    } else if (msg->data == "disable") {
      for (int dxl_id = DXL_MIN_ID; dxl_id <= DXL_MAX_ID; ++dxl_id) disableTorque(dxl_id);
    } else {
      RCLCPP_WARN(this->get_logger(), "Unknown command: %s", msg->data.c_str());
    }
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamixelController>());
  rclcpp::shutdown();
  return 0;
}
