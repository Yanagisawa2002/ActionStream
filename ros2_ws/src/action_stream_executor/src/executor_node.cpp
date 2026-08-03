#include "action_stream_executor/executor_node.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace action_stream_executor
{
namespace
{

[[nodiscard]] rclcpp::QoS reliable_qos(const std::size_t depth)
{
  return rclcpp::QoS(rclcpp::KeepLast(depth)).reliable();
}

}  // namespace

ActionStreamExecutorNode::ActionStreamExecutorNode(const rclcpp::NodeOptions & options)
: Node("action_stream_executor", options)
{
  const auto strategy_name = declare_parameter<std::string>("strategy", "aligned_async");
  const auto action_dimension_parameter = declare_parameter<std::int64_t>("action_dimension", 7);
  if (action_dimension_parameter != static_cast<std::int64_t>(kCanonicalActionDimension)) {
    throw std::invalid_argument(
            "action_dimension must remain 7 for canonical absolute ActionStream commands");
  }
  const auto action_dimension = static_cast<std::size_t>(action_dimension_parameter);
  const auto safe_hold_command = declare_parameter<std::vector<double>>(
    "safe_hold_command", std::vector<double>{});
  if (safe_hold_command.empty()) {
    throw std::invalid_argument(
            "safe_hold_command is required; provide a finite nonzero 7D absolute command, "
            "for example [0.45, 0, 0.35, 3.141592653589793, 0, 0, 1]");
  }
  state_machine_ = std::make_unique<ExecutorStateMachine>(
    parse_strategy(strategy_name), action_dimension, safe_hold_command);

  const auto observation_topic =
    declare_parameter<std::string>("observation_topic", "observation");
  const auto request_topic =
    declare_parameter<std::string>("inference_request_topic", "inference_request");
  const auto chunk_topic =
    declare_parameter<std::string>("action_chunk_topic", "action_chunk");
  const auto command_topic =
    declare_parameter<std::string>("robot_command_topic", "robot_command");
  const auto diagnostics_topic =
    declare_parameter<std::string>("diagnostics_topic", "executor_diagnostics");
  const auto event_topic =
    declare_parameter<std::string>("runtime_event_topic", "runtime_event");
  const auto episode_control_topic =
    declare_parameter<std::string>("episode_control_topic", "episode_control");

  observation_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  inference_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  lifecycle_callback_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions observation_options;
  observation_options.callback_group = observation_callback_group_;
  observation_subscription_ = create_subscription<ObservationMsg>(
    observation_topic, reliable_qos(50U),
    [this](const ObservationMsg::ConstSharedPtr message) {on_observation(message);},
    observation_options);

  rclcpp::SubscriptionOptions inference_options;
  inference_options.callback_group = inference_callback_group_;
  request_subscription_ = create_subscription<RequestMsg>(
    request_topic, reliable_qos(100U),
    [this](const RequestMsg::ConstSharedPtr message) {on_request(message);},
    inference_options);
  chunk_subscription_ = create_subscription<ChunkMsg>(
    chunk_topic, reliable_qos(100U),
    [this](const ChunkMsg::ConstSharedPtr message) {on_chunk(message);},
    inference_options);

  rclcpp::SubscriptionOptions lifecycle_options;
  lifecycle_options.callback_group = lifecycle_callback_group_;
  episode_control_subscription_ = create_subscription<EpisodeControlMsg>(
    episode_control_topic, reliable_qos(10U).transient_local(),
    [this](const EpisodeControlMsg::ConstSharedPtr message) {on_episode_control(message);},
    lifecycle_options);

  command_publisher_ =
    create_publisher<action_stream_msgs::msg::RobotCommand>(command_topic, reliable_qos(50U));
  diagnostics_publisher_ =
    create_publisher<action_stream_msgs::msg::ExecutorDiagnostics>(
    diagnostics_topic, reliable_qos(50U));
  event_publisher_ =
    create_publisher<action_stream_msgs::msg::RuntimeEvent>(event_topic, reliable_qos(200U));

  RCLCPP_INFO(
    get_logger(), "ActionStream executor ready: strategy=%s action_dimension=%zu",
    strategy_name.c_str(), action_dimension);
}

void ActionStreamExecutorNode::on_episode_control(
  const EpisodeControlMsg::ConstSharedPtr & message)
{
  if (!should_process_episode_control(message->message_kind)) {
    if (message->message_kind == EpisodeControlMsg::STATUS) {
      RCLCPP_DEBUG(
        get_logger(), "Ignoring EpisodeControl status/ACK for episode=%s command=%u",
        message->episode_id.c_str(), static_cast<unsigned int>(message->command));
    } else {
      RCLCPP_WARN(
        get_logger(), "Ignoring EpisodeControl with unknown message_kind=%u",
        static_cast<unsigned int>(message->message_kind));
    }
    return;
  }
  const ClockStamp stamp{
    time_to_nanoseconds(message->sim_stamp), message->steady_time_ns, message->wall_time_ns};
  Decision decision;
  if (message->command == EpisodeControlMsg::START) {
    decision = state_machine_->start_episode(
      message->episode_id, message->generation_id, 0U, stamp);
  } else if (message->command == EpisodeControlMsg::RESET) {
    decision = state_machine_->reset_episode(
      message->episode_id, message->generation_id, 0U, stamp);
  } else if (message->command == EpisodeControlMsg::TERMINATE) {
    decision = state_machine_->terminate_episode(message->episode_id, message->success, stamp);
  } else {
    RCLCPP_ERROR(
      get_logger(), "Ignoring EpisodeControl with unknown command=%u",
      static_cast<unsigned int>(message->command));
    return;
  }
  if (!decision.accepted) {
    RCLCPP_WARN(
      get_logger(), "EpisodeControl rejected for episode=%s: %s",
      message->episode_id.c_str(), decision.reason.c_str());
  }
  publish_state_outputs(stamp);
}

void ActionStreamExecutorNode::on_observation(const ObservationMsg::ConstSharedPtr & message)
{
  const ClockStamp stamp{
    time_to_nanoseconds(message->sim_stamp), message->steady_time_ns, message->wall_time_ns};
  const auto decision = state_machine_->observe(
    ObservationRecord{stamp, message->episode_id, message->observation_step, message->terminated});
  if (decision.accepted && !message->terminated) {
    if (message->observation_step == std::numeric_limits<std::uint64_t>::max()) {
      RCLCPP_ERROR(get_logger(), "Observation step overflow for episode=%s", message->episode_id.c_str());
    } else {
      const auto command = state_machine_->command_for_step(
        message->episode_id, message->observation_step + 1U,
        ClockStamp{stamp.sim_time_ns, steady_now_ns(), wall_now_ns()});
      if (command.has_value()) {
        publish_command(command.value());
      }
    }
  }
  publish_state_outputs(stamp);
}

void ActionStreamExecutorNode::on_request(const RequestMsg::ConstSharedPtr & message)
{
  const ClockStamp stamp{
    time_to_nanoseconds(message->sim_stamp), message->request_steady_time_ns,
    message->request_wall_time_ns};
  if (message->observation.episode_id != message->episode_id ||
    message->observation.observation_step != message->source_observation_step)
  {
    RCLCPP_WARN(
      get_logger(), "InferenceRequest %llu rejected: embedded_observation_mismatch",
      static_cast<unsigned long long>(message->request_id));
    return;
  }

  // ROS preserves order within one topic, not across the observation and
  // request topics.  The request carries the exact source observation so the
  // executor can deterministically close that cross-topic race before
  // registering its provenance.  Command dispatch remains owned by the
  // observation callback; this synchronization never executes a command.
  const auto before = state_machine_->diagnostics(steady_now_ns());
  if (message->source_observation_step > before.latest_observation_step) {
    const auto & observation = message->observation;
    const auto observation_decision = state_machine_->observe(
      ObservationRecord{
        ClockStamp{
          time_to_nanoseconds(observation.sim_stamp), observation.steady_time_ns,
          observation.wall_time_ns},
        observation.episode_id, observation.observation_step, observation.terminated});
    if (!observation_decision.accepted) {
      RCLCPP_WARN(
        get_logger(), "InferenceRequest %llu source observation rejected: %s",
        static_cast<unsigned long long>(message->request_id),
        observation_decision.reason.c_str());
    }
  }
  const auto decision = state_machine_->register_request(
    RequestRecord{
      stamp, message->episode_id, message->request_id, message->generation_id,
      message->source_observation_step, time_to_nanoseconds(message->source_sim_stamp),
      message->source_observation_steady_time_ns, message->expected_horizon});
  if (!decision.accepted) {
    RCLCPP_WARN(
      get_logger(), "InferenceRequest %llu rejected: %s",
      static_cast<unsigned long long>(message->request_id), decision.reason.c_str());
  }
  publish_state_outputs(stamp);
}

void ActionStreamExecutorNode::on_chunk(const ChunkMsg::ConstSharedPtr & message)
{
  ChunkRecord chunk;
  chunk.stamp = ClockStamp{
    time_to_nanoseconds(message->sim_stamp), message->steady_time_ns, message->wall_time_ns};
  chunk.episode_id = message->episode_id;
  chunk.request_id = message->request_id;
  chunk.generation_id = message->generation_id;
  chunk.source_observation_step = message->source_observation_step;
  chunk.source_sim_time_ns = time_to_nanoseconds(message->source_sim_stamp);
  chunk.source_observation_steady_time_ns = message->source_observation_steady_time_ns;
  chunk.action_dimension = message->action_dimension;
  chunk.inference_complete_steady_time_ns = message->inference_complete_steady_time_ns;
  chunk.response_publish_steady_time_ns = message->response_publish_steady_time_ns;
  chunk.actions.reserve(message->actions.size());
  for (const auto & action : message->actions) {
    chunk.actions.push_back(TargetActionRecord{action.target_step, action.command});
  }

  const auto decision = state_machine_->ingest_chunk(chunk);
  if (!decision.accepted) {
    RCLCPP_WARN(
      get_logger(), "ActionChunk request=%llu generation=%llu rejected: %s",
      static_cast<unsigned long long>(message->request_id),
      static_cast<unsigned long long>(message->generation_id), decision.reason.c_str());
  }
  publish_state_outputs(chunk.stamp);
}

void ActionStreamExecutorNode::publish_command(const CommandRecord & command)
{
  action_stream_msgs::msg::RobotCommand message;
  message.sim_stamp = nanoseconds_to_time(command.stamp.sim_time_ns);
  message.steady_time_ns = command.stamp.steady_time_ns;
  message.wall_time_ns = command.stamp.wall_time_ns;
  message.episode_id = command.episode_id;
  message.actual_target_step = command.actual_target_step;
  message.source_request_id = command.source_request_id;
  message.source_generation_id = command.source_generation_id;
  message.source_observation_step = command.source_observation_step;
  message.source_target_step = command.source_target_step;
  message.command = command.command;
  message.hold = command.hold;
  message.reason = command.reason;
  command_publisher_->publish(message);
}

void ActionStreamExecutorNode::publish_state_outputs(const ClockStamp & stamp)
{
  for (const auto & event : state_machine_->drain_events()) {
    publish_event(event);
  }

  const auto diagnostics = state_machine_->diagnostics(stamp.steady_time_ns);
  action_stream_msgs::msg::ExecutorDiagnostics message;
  message.sim_stamp = nanoseconds_to_time(stamp.sim_time_ns);
  message.steady_time_ns = stamp.steady_time_ns;
  message.wall_time_ns = stamp.wall_time_ns;
  message.strategy = to_string(diagnostics.strategy);
  message.episode_id = diagnostics.episode_id;
  message.episode_active = diagnostics.episode_active;
  message.episode_terminated = diagnostics.episode_terminated;
  message.episode_success = diagnostics.episode_success;
  message.latest_observation_step = diagnostics.latest_observation_step;
  message.has_executed_target_step = diagnostics.has_executed_target_step;
  message.latest_executed_target_step = diagnostics.latest_executed_target_step;
  message.active_generation_id = diagnostics.active_generation_id;
  message.queue_length = static_cast<std::uint64_t>(diagnostics.queue_length);
  message.accepted_chunks = diagnostics.accepted_chunks;
  message.rejected_chunks = diagnostics.rejected_chunks;
  message.rejected_previous_episode_chunks = diagnostics.rejected_previous_episode_chunks;
  message.rejected_stale_generation_chunks = diagnostics.rejected_stale_generation_chunks;
  message.rejected_unknown_request_chunks = diagnostics.rejected_unknown_request_chunks;
  message.duplicate_responses = diagnostics.duplicate_responses;
  message.expired_actions_removed = diagnostics.expired_actions_removed;
  message.duplicate_actions_removed = diagnostics.duplicate_actions_removed;
  message.queue_rebuilds = diagnostics.queue_rebuilds;
  message.deadline_misses = diagnostics.deadline_misses;
  message.hold_steps = diagnostics.hold_steps;
  message.total_hold_duration_ns = diagnostics.total_hold_duration_ns;
  message.current_hold_duration_ns = diagnostics.current_hold_duration_ns;
  message.current_action_age_steps = diagnostics.current_action_age_steps;
  message.executed_source_generation_id = diagnostics.executed_source_generation_id;
  diagnostics_publisher_->publish(message);
}

void ActionStreamExecutorNode::publish_event(const RuntimeEventRecord & event)
{
  action_stream_msgs::msg::RuntimeEvent message;
  message.sim_stamp = nanoseconds_to_time(event.stamp.sim_time_ns);
  message.steady_time_ns = event.stamp.steady_time_ns;
  message.wall_time_ns = event.stamp.wall_time_ns;
  message.event_type = event.event_type;
  message.reason = event.reason;
  message.strategy = to_string(event.strategy);
  message.episode_id = event.episode_id;
  message.request_id = event.request_id;
  message.generation_id = event.generation_id;
  message.source_observation_step = event.source_observation_step;
  message.source_target_step = event.source_target_step;
  message.actual_target_step = event.actual_target_step;
  message.active_generation_id = event.active_generation_id;
  message.queue_length_before = static_cast<std::uint64_t>(event.queue_length_before);
  message.queue_length_after = static_cast<std::uint64_t>(event.queue_length_after);
  message.action_count = static_cast<std::uint64_t>(event.action_count);
  message.detail = event.detail;
  event_publisher_->publish(message);
}

std::int64_t ActionStreamExecutorNode::time_to_nanoseconds(
  const builtin_interfaces::msg::Time & time)
{
  return combine_time_ns(time.sec, time.nanosec);
}

builtin_interfaces::msg::Time ActionStreamExecutorNode::nanoseconds_to_time(
  const std::int64_t value)
{
  const auto parts = split_time_ns(value);
  builtin_interfaces::msg::Time time;
  time.sec = parts.seconds;
  time.nanosec = parts.nanoseconds;
  return time;
}

std::uint64_t ActionStreamExecutorNode::steady_now_ns()
{
  const auto value = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(value).count());
}

std::uint64_t ActionStreamExecutorNode::wall_now_ns()
{
  const auto value = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(value).count());
}

}  // namespace action_stream_executor
