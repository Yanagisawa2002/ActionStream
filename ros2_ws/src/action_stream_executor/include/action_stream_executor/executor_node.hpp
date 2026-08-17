#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "action_stream_executor/executor_state_machine.hpp"
#include "action_stream_msgs/msg/action_chunk.hpp"
#include "action_stream_msgs/msg/episode_control.hpp"
#include "action_stream_msgs/msg/executor_diagnostics.hpp"
#include "action_stream_msgs/msg/inference_request.hpp"
#include "action_stream_msgs/msg/observation.hpp"
#include "action_stream_msgs/msg/robot_command.hpp"
#include "action_stream_msgs/msg/runtime_event.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "rclcpp/rclcpp.hpp"

namespace action_stream_executor
{

class ActionStreamExecutorNode final : public rclcpp::Node
{
public:
  explicit ActionStreamExecutorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using ObservationMsg = action_stream_msgs::msg::Observation;
  using RequestMsg = action_stream_msgs::msg::InferenceRequest;
  using ChunkMsg = action_stream_msgs::msg::ActionChunk;
  using EpisodeControlMsg = action_stream_msgs::msg::EpisodeControl;

  void on_observation(const ObservationMsg::ConstSharedPtr & message);
  void on_request(const RequestMsg::ConstSharedPtr & message);
  void on_chunk(const ChunkMsg::ConstSharedPtr & message);
  void on_episode_control(const EpisodeControlMsg::ConstSharedPtr & message);
  void publish_command(const CommandRecord & command);
  void publish_state_outputs(const ClockStamp & stamp);
  void publish_event(const RuntimeEventRecord & event);

  [[nodiscard]] static std::int64_t time_to_nanoseconds(
    const builtin_interfaces::msg::Time & time);
  [[nodiscard]] static builtin_interfaces::msg::Time nanoseconds_to_time(std::int64_t value);
  [[nodiscard]] static std::uint64_t steady_now_ns();
  [[nodiscard]] static std::uint64_t wall_now_ns();

  std::unique_ptr<ExecutorStateMachine> state_machine_;

  rclcpp::CallbackGroup::SharedPtr observation_callback_group_;
  rclcpp::CallbackGroup::SharedPtr inference_callback_group_;
  rclcpp::CallbackGroup::SharedPtr lifecycle_callback_group_;
  rclcpp::Subscription<ObservationMsg>::SharedPtr observation_subscription_;
  rclcpp::Subscription<RequestMsg>::SharedPtr request_subscription_;
  rclcpp::Subscription<ChunkMsg>::SharedPtr chunk_subscription_;
  rclcpp::Subscription<EpisodeControlMsg>::SharedPtr episode_control_subscription_;
  rclcpp::Publisher<action_stream_msgs::msg::RobotCommand>::SharedPtr command_publisher_;
  rclcpp::Publisher<action_stream_msgs::msg::ExecutorDiagnostics>::SharedPtr
  diagnostics_publisher_;
  rclcpp::Publisher<action_stream_msgs::msg::RuntimeEvent>::SharedPtr event_publisher_;
};

}  // namespace action_stream_executor
