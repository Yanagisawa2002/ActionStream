#include "action_stream_executor/executor_state_machine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace action_stream_executor
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

[[nodiscard]] bool add_would_overflow(std::uint64_t value, std::uint64_t increment)
{
  return increment > std::numeric_limits<std::uint64_t>::max() - value;
}

}  // namespace

std::int64_t combine_time_ns(const std::int32_t seconds, const std::uint32_t nanoseconds)
{
  if (nanoseconds >= static_cast<std::uint32_t>(kNanosecondsPerSecond)) {
    throw std::invalid_argument("nanoseconds must be less than one second");
  }
  return static_cast<std::int64_t>(seconds) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(nanoseconds);
}

TimeParts split_time_ns(const std::int64_t nanoseconds)
{
  auto seconds = nanoseconds / kNanosecondsPerSecond;
  auto remainder = nanoseconds % kNanosecondsPerSecond;
  if (remainder < 0) {
    remainder += kNanosecondsPerSecond;
    --seconds;
  }
  if (seconds < std::numeric_limits<std::int32_t>::min() ||
    seconds > std::numeric_limits<std::int32_t>::max())
  {
    throw std::out_of_range("timestamp is outside builtin_interfaces/Time range");
  }
  return TimeParts{
    static_cast<std::int32_t>(seconds), static_cast<std::uint32_t>(remainder)};
}

Strategy parse_strategy(const std::string_view value)
{
  if (value == "sync_hold") {
    return Strategy::kSyncHold;
  }
  if (value == "naive_async") {
    return Strategy::kNaiveAsync;
  }
  if (value == "aligned_async") {
    return Strategy::kAlignedAsync;
  }
  throw std::invalid_argument(
          "strategy must be one of sync_hold, naive_async, aligned_async");
}

std::string to_string(const Strategy strategy)
{
  switch (strategy) {
    case Strategy::kSyncHold:
      return "sync_hold";
    case Strategy::kNaiveAsync:
      return "naive_async";
    case Strategy::kAlignedAsync:
      return "aligned_async";
  }
  throw std::logic_error("unknown ActionStream strategy");
}

bool should_process_episode_control(const std::uint8_t message_kind) noexcept
{
  return message_kind == kEpisodeControlRequest;
}

void validate_safe_hold_command(
  const std::vector<double> & command, const std::size_t expected_dimension,
  const bool reject_all_zero)
{
  if (expected_dimension == 0U) {
    throw std::invalid_argument("expected safe-hold dimension must be positive");
  }
  if (command.size() != expected_dimension) {
    throw std::invalid_argument("safe_hold_command does not match action_dimension");
  }
  if (!std::all_of(
      command.begin(), command.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument("safe_hold_command must contain only finite values");
  }
  if (reject_all_zero && std::all_of(
      command.begin(), command.end(),
      [](const double value) {return value == 0.0;}))
  {
    throw std::invalid_argument(
            "safe_hold_command must not be all zero for absolute commands");
  }
}

ExecutorStateMachine::ExecutorStateMachine(
  const Strategy strategy, const std::size_t action_dimension,
  std::vector<double> safe_hold_command)
: strategy_(strategy),
  action_dimension_(action_dimension),
  safe_hold_command_(std::move(safe_hold_command))
{
  if (action_dimension_ == 0U) {
    throw std::invalid_argument("action_dimension must be positive");
  }
  validate_safe_hold_command(
    safe_hold_command_, action_dimension_, action_dimension_ == kCanonicalActionDimension);
}

Decision ExecutorStateMachine::start_episode(
  const std::string & episode_id, const std::uint64_t initial_generation_id,
  const std::uint64_t initial_observation_step, const ClockStamp & stamp)
{
  std::scoped_lock lock(mutex_);
  if (episode_active_) {
    RuntimeEventRecord event;
    event.stamp = stamp;
    event.event_type = "episode_rejected";
    event.reason = "episode_already_active";
    event.detail = "terminate or reset the active episode before starting another";
    push_event_locked(std::move(event));
    return {false, "episode_already_active", 0U, 0U, 0U, queue_.size()};
  }
  return begin_episode_locked(
    "episode_started", episode_id, initial_generation_id, initial_observation_step, stamp);
}

Decision ExecutorStateMachine::reset_episode(
  const std::string & episode_id, const std::uint64_t initial_generation_id,
  const std::uint64_t initial_observation_step, const ClockStamp & stamp)
{
  std::scoped_lock lock(mutex_);
  close_hold_locked(stamp.steady_time_ns);
  return begin_episode_locked(
    "episode_reset", episode_id, initial_generation_id, initial_observation_step, stamp);
}

Decision ExecutorStateMachine::begin_episode_locked(
  const std::string & event_type, const std::string & episode_id,
  const std::uint64_t initial_generation_id, const std::uint64_t initial_observation_step,
  const ClockStamp & stamp)
{
  if (episode_id.empty()) {
    return {false, "empty_episode_id", 0U, 0U, 0U, queue_.size()};
  }

  episode_id_ = episode_id;
  episode_active_ = true;
  episode_terminated_ = false;
  episode_success_ = false;
  latest_observation_step_ = initial_observation_step;
  latest_executed_target_step_.reset();
  active_generation_id_ = initial_generation_id;
  queue_.clear();
  requests_.clear();
  completed_request_ids_.clear();
  latest_accepted_plan_key_.reset();
  sync_request_in_flight_.reset();
  last_policy_action_.reset();
  last_command_.clear();
  hold_started_steady_time_ns_.reset();
  events_.clear();
  reset_counters_locked();

  RuntimeEventRecord event;
  event.stamp = stamp;
  event.event_type = event_type;
  event.reason = "accepted";
  event.generation_id = initial_generation_id;
  event.actual_target_step = initial_observation_step;
  push_event_locked(std::move(event));
  return {true, "accepted", 0U, 0U, 0U, 0U};
}

Decision ExecutorStateMachine::terminate_episode(
  const std::string & episode_id, const bool success, const ClockStamp & stamp)
{
  std::scoped_lock lock(mutex_);
  if (episode_id != episode_id_) {
    RuntimeEventRecord event;
    event.stamp = stamp;
    event.event_type = "episode_rejected";
    event.reason = "previous_episode";
    event.episode_id = episode_id;
    push_event_locked(std::move(event));
    return {false, "previous_episode", 0U, 0U, 0U, queue_.size()};
  }
  if (!episode_active_) {
    return {false, "episode_not_active", 0U, 0U, 0U, queue_.size()};
  }

  close_hold_locked(stamp.steady_time_ns);
  const auto queue_before = queue_.size();
  queue_.clear();
  requests_.clear();
  completed_request_ids_.clear();
  latest_accepted_plan_key_.reset();
  sync_request_in_flight_.reset();
  episode_active_ = false;
  episode_terminated_ = true;
  episode_success_ = success;

  const std::string termination_reason = success ? "success" : "terminated";
  RuntimeEventRecord event;
  event.stamp = stamp;
  event.event_type = "episode_terminated";
  event.reason = termination_reason;
  event.queue_length_before = queue_before;
  event.queue_length_after = 0U;
  push_event_locked(std::move(event));
  return {true, termination_reason, 0U, 0U, 0U, 0U};
}

Decision ExecutorStateMachine::observe(const ObservationRecord & observation)
{
  std::scoped_lock lock(mutex_);
  if (observation.episode_id != episode_id_) {
    RuntimeEventRecord event;
    event.stamp = observation.stamp;
    event.event_type = "observation_rejected";
    event.reason = "previous_episode";
    event.episode_id = observation.episode_id;
    event.actual_target_step = observation.observation_step;
    push_event_locked(std::move(event));
    return {false, "previous_episode", 0U, 0U, 0U, queue_.size()};
  }
  if (!episode_active_) {
    return {false, "episode_not_active", 0U, 0U, 0U, queue_.size()};
  }
  if (observation.observation_step < latest_observation_step_) {
    RuntimeEventRecord event;
    event.stamp = observation.stamp;
    event.event_type = "observation_rejected";
    event.reason = "non_monotonic_observation_step";
    event.actual_target_step = observation.observation_step;
    push_event_locked(std::move(event));
    return {false, "non_monotonic_observation_step", 0U, 0U, 0U, queue_.size()};
  }

  latest_observation_step_ = observation.observation_step;
  RuntimeEventRecord event;
  event.stamp = observation.stamp;
  event.event_type = "observation_received";
  event.reason = observation.terminated ? "terminal_observation" : "accepted";
  event.actual_target_step = observation.observation_step;
  event.queue_length_before = queue_.size();
  event.queue_length_after = queue_.size();
  push_event_locked(std::move(event));

  if (observation.terminated) {
    close_hold_locked(observation.stamp.steady_time_ns);
    queue_.clear();
    requests_.clear();
    completed_request_ids_.clear();
    latest_accepted_plan_key_.reset();
    sync_request_in_flight_.reset();
    episode_active_ = false;
    episode_terminated_ = true;
    episode_success_ = false;
    return {true, "terminal_observation", 0U, 0U, 0U, 0U};
  }
  return {true, "accepted", 0U, 0U, 0U, queue_.size()};
}

Decision ExecutorStateMachine::register_request(const RequestRecord & request)
{
  std::scoped_lock lock(mutex_);
  const auto reject = [this, &request](const std::string & reason) {
      RuntimeEventRecord event;
      event.stamp = request.stamp;
      event.event_type = "request_rejected";
      event.reason = reason;
      event.episode_id = request.episode_id;
      event.request_id = request.request_id;
      event.generation_id = request.generation_id;
      event.source_observation_step = request.source_observation_step;
      event.queue_length_before = queue_.size();
      event.queue_length_after = queue_.size();
      push_event_locked(std::move(event));
      return Decision{false, reason, 0U, 0U, 0U, queue_.size()};
    };

  if (request.episode_id != episode_id_) {
    return reject("previous_episode");
  }
  if (!episode_active_) {
    return reject(episode_terminated_ ? "episode_terminated" : "episode_not_active");
  }
  if (request.expected_horizon == 0U) {
    return reject("zero_horizon");
  }
  if (request.source_observation_step > latest_observation_step_) {
    return reject("future_source_observation");
  }
  if (requests_.count(request.request_id) != 0U ||
    completed_request_ids_.count(request.request_id) != 0U)
  {
    return reject("duplicate_request_id");
  }
  if (strategy_ != Strategy::kNaiveAsync && request.generation_id < active_generation_id_) {
    return reject("stale_generation");
  }
  if (strategy_ == Strategy::kSyncHold) {
    if (sync_request_in_flight_.has_value()) {
      return reject("sync_request_already_in_flight");
    }
    if (!queue_.empty()) {
      return reject("sync_queue_not_empty");
    }
    sync_request_in_flight_ = request.request_id;
  }

  if (request.generation_id > active_generation_id_) {
    active_generation_id_ = request.generation_id;
    latest_accepted_plan_key_.reset();
  }
  requests_.emplace(request.request_id, request);
  RuntimeEventRecord event;
  event.stamp = request.stamp;
  event.event_type = "request_registered";
  event.reason = "accepted";
  event.request_id = request.request_id;
  event.generation_id = request.generation_id;
  event.source_observation_step = request.source_observation_step;
  event.queue_length_before = queue_.size();
  event.queue_length_after = queue_.size();
  push_event_locked(std::move(event));
  return {true, "accepted", 0U, 0U, 0U, queue_.size()};
}

bool ExecutorStateMachine::validate_command_locked(const TargetActionRecord & action) const
{
  return action.command.size() == action_dimension_ &&
         std::all_of(
    action.command.begin(), action.command.end(),
    [](const double value) {return std::isfinite(value);});
}

std::vector<ExecutorStateMachine::QueuedAction> ExecutorStateMachine::make_valid_actions_locked(
  const ChunkRecord & chunk, const RequestRecord & request, std::size_t * const expired,
  std::size_t * const duplicates, std::string * const error) const
{
  std::vector<QueuedAction> valid;
  std::unordered_set<std::uint64_t> seen_targets;
  *expired = 0U;
  *duplicates = 0U;
  error->clear();

  if (chunk.action_dimension != action_dimension_) {
    *error = "action_dimension_mismatch";
    return valid;
  }
  if (chunk.actions.size() != request.expected_horizon) {
    *error = "horizon_mismatch";
    return valid;
  }
  if (add_would_overflow(request.source_observation_step, request.expected_horizon)) {
    *error = "target_step_overflow";
    return valid;
  }
  const auto first_target = request.source_observation_step + 1U;
  const auto last_target = request.source_observation_step + request.expected_horizon;
  const auto insertion_step =
    latest_observation_step_ == std::numeric_limits<std::uint64_t>::max() ?
    latest_observation_step_ : latest_observation_step_ + 1U;

  for (const auto & action : chunk.actions) {
    if (!validate_command_locked(action)) {
      *error = "invalid_command";
      return {};
    }
    if (action.target_step < first_target || action.target_step > last_target) {
      *error = "target_outside_horizon";
      return {};
    }
    if (action.target_step < insertion_step ||
      (latest_executed_target_step_.has_value() &&
      action.target_step <= latest_executed_target_step_.value()))
    {
      ++(*expired);
      continue;
    }
    if (!seen_targets.insert(action.target_step).second) {
      ++(*duplicates);
      continue;
    }
    valid.push_back(
      QueuedAction{
        chunk.request_id, chunk.generation_id, chunk.source_observation_step,
        action.target_step, action.command});
  }

  std::sort(
    valid.begin(), valid.end(),
    [](const QueuedAction & left, const QueuedAction & right) {
      return left.source_target_step < right.source_target_step;
    });
  return valid;
}

Decision ExecutorStateMachine::reject_chunk_locked(
  const ChunkRecord & chunk, const std::string & reason, const std::size_t queue_before,
  const std::string & detail)
{
  ++rejected_chunks_;
  if (reason == "previous_episode") {
    ++rejected_previous_episode_chunks_;
  } else if (reason == "stale_generation" || reason == "future_generation") {
    ++rejected_stale_generation_chunks_;
  } else if (reason == "unknown_request") {
    ++rejected_unknown_request_chunks_;
  } else if (reason == "duplicate_response") {
    ++duplicate_responses_;
  }

  RuntimeEventRecord event;
  event.stamp = chunk.stamp;
  event.event_type = "chunk_rejected";
  event.reason = reason;
  event.episode_id = chunk.episode_id;
  event.request_id = chunk.request_id;
  event.generation_id = chunk.generation_id;
  event.source_observation_step = chunk.source_observation_step;
  event.queue_length_before = queue_before;
  event.queue_length_after = queue_.size();
  event.action_count = chunk.actions.size();
  event.detail = detail;
  push_event_locked(std::move(event));
  return {false, reason, 0U, 0U, 0U, queue_.size()};
}

Decision ExecutorStateMachine::ingest_chunk(const ChunkRecord & chunk)
{
  std::scoped_lock lock(mutex_);
  const auto queue_before = queue_.size();
  if (chunk.episode_id != episode_id_) {
    return reject_chunk_locked(chunk, "previous_episode", queue_before);
  }
  if (!episode_active_) {
    return reject_chunk_locked(
      chunk, episode_terminated_ ? "episode_terminated" : "episode_not_active", queue_before);
  }
  if (strategy_ != Strategy::kNaiveAsync && chunk.generation_id != active_generation_id_) {
    return reject_chunk_locked(
      chunk,
      chunk.generation_id < active_generation_id_ ? "stale_generation" : "future_generation",
      queue_before);
  }
  if (completed_request_ids_.count(chunk.request_id) != 0U) {
    return reject_chunk_locked(chunk, "duplicate_response", queue_before);
  }
  const auto request_iterator = requests_.find(chunk.request_id);
  if (request_iterator == requests_.end()) {
    return reject_chunk_locked(chunk, "unknown_request", queue_before);
  }
  const auto & request = request_iterator->second;
  if (request.episode_id != chunk.episode_id ||
    request.generation_id != chunk.generation_id ||
    request.source_observation_step != chunk.source_observation_step ||
    request.source_sim_time_ns != chunk.source_sim_time_ns ||
    request.source_observation_steady_time_ns != chunk.source_observation_steady_time_ns)
  {
    return reject_chunk_locked(chunk, "provenance_mismatch", queue_before);
  }
  if (strategy_ == Strategy::kAlignedAsync && latest_accepted_plan_key_.has_value()) {
    const auto incoming_key = std::make_pair(
      chunk.source_observation_step, chunk.request_id);
    if (incoming_key <= latest_accepted_plan_key_.value()) {
      const auto & latest_key = latest_accepted_plan_key_.value();
      return reject_chunk_locked(
        chunk, "stale_plan_freshness", queue_before,
        "incoming_source_step=" + std::to_string(incoming_key.first) +
        ",incoming_request_id=" + std::to_string(incoming_key.second) +
        ",latest_source_step=" + std::to_string(latest_key.first) +
        ",latest_request_id=" + std::to_string(latest_key.second));
    }
  }
  if (chunk.inference_complete_steady_time_ns > chunk.response_publish_steady_time_ns) {
    return reject_chunk_locked(chunk, "non_monotonic_response_timestamps", queue_before);
  }
  if (strategy_ == Strategy::kSyncHold &&
    (!sync_request_in_flight_.has_value() ||
    sync_request_in_flight_.value() != chunk.request_id))
  {
    return reject_chunk_locked(chunk, "unexpected_sync_response", queue_before);
  }

  if (chunk.action_dimension != action_dimension_) {
    return reject_chunk_locked(chunk, "action_dimension_mismatch", queue_before);
  }
  if (chunk.actions.size() != request.expected_horizon) {
    return reject_chunk_locked(chunk, "horizon_mismatch", queue_before);
  }
  if (add_would_overflow(request.source_observation_step, request.expected_horizon)) {
    return reject_chunk_locked(chunk, "target_step_overflow", queue_before);
  }
  const auto first_target = request.source_observation_step + 1U;
  const auto last_target = request.source_observation_step + request.expected_horizon;
  for (const auto & action : chunk.actions) {
    if (!validate_command_locked(action)) {
      return reject_chunk_locked(chunk, "invalid_command", queue_before);
    }
    if (action.target_step < first_target || action.target_step > last_target) {
      return reject_chunk_locked(chunk, "target_outside_horizon", queue_before);
    }
  }

  if (strategy_ == Strategy::kNaiveAsync) {
    for (const auto & action : chunk.actions) {
      queue_.push_back(
        QueuedAction{
          chunk.request_id, chunk.generation_id, chunk.source_observation_step,
          action.target_step, action.command});
    }
    completed_request_ids_.insert(chunk.request_id);
    requests_.erase(request_iterator);
    ++accepted_chunks_;

    RuntimeEventRecord event;
    event.stamp = chunk.stamp;
    event.event_type = "queue_updated";
    event.reason = "arrival_order_append";
    event.request_id = chunk.request_id;
    event.generation_id = chunk.generation_id;
    event.source_observation_step = chunk.source_observation_step;
    event.queue_length_before = queue_before;
    event.queue_length_after = queue_.size();
    event.action_count = chunk.actions.size();
    push_event_locked(std::move(event));
    return {true, "arrival_order_append", chunk.actions.size(), 0U, 0U, queue_.size()};
  }

  std::size_t expired = 0U;
  std::size_t duplicates = 0U;
  std::string validation_error;
  auto valid = make_valid_actions_locked(
    chunk, request, &expired, &duplicates, &validation_error);
  if (!validation_error.empty()) {
    return reject_chunk_locked(chunk, validation_error, queue_before);
  }
  expired_actions_removed_ += expired;
  duplicate_actions_removed_ += duplicates;
  if (valid.empty()) {
    completed_request_ids_.insert(chunk.request_id);
    requests_.erase(request_iterator);
    if (strategy_ == Strategy::kSyncHold) {
      sync_request_in_flight_.reset();
    }
    auto decision = reject_chunk_locked(chunk, "fully_expired", queue_before);
    decision.actions_expired = expired;
    decision.duplicates_removed = duplicates;
    return decision;
  }

  std::deque<QueuedAction> replacement(valid.begin(), valid.end());
  queue_.swap(replacement);
  completed_request_ids_.insert(chunk.request_id);
  requests_.erase(request_iterator);
  if (strategy_ == Strategy::kSyncHold) {
    sync_request_in_flight_.reset();
  }
  ++accepted_chunks_;
  ++queue_rebuilds_;
  if (strategy_ == Strategy::kAlignedAsync) {
    latest_accepted_plan_key_ = std::make_pair(
      chunk.source_observation_step, chunk.request_id);
  }

  const std::string update_reason =
    strategy_ == Strategy::kSyncHold ? "sync_valid_rebuild" : "aligned_atomic_rebuild";
  RuntimeEventRecord event;
  event.stamp = chunk.stamp;
  event.event_type = "queue_updated";
  event.reason = update_reason;
  event.request_id = chunk.request_id;
  event.generation_id = chunk.generation_id;
  event.source_observation_step = chunk.source_observation_step;
  event.queue_length_before = queue_before;
  event.queue_length_after = queue_.size();
  event.action_count = queue_.size();
  event.detail = "expired=" + std::to_string(expired) +
    ",duplicates=" + std::to_string(duplicates);
  push_event_locked(std::move(event));
  return {
    true, update_reason, queue_.size(), expired, duplicates, queue_.size()};
}

CommandRecord ExecutorStateMachine::make_hold_locked(
  const std::uint64_t actual_target_step, const ClockStamp & stamp, const std::string & reason)
{
  if (!hold_started_steady_time_ns_.has_value()) {
    hold_started_steady_time_ns_ = stamp.steady_time_ns;
  }
  ++hold_steps_;
  ++deadline_misses_;

  CommandRecord command;
  command.stamp = stamp;
  command.episode_id = episode_id_;
  command.actual_target_step = actual_target_step;
  command.command = last_command_.empty() ? safe_hold_command_ : last_command_;
  command.hold = true;
  command.reason = reason;
  if (last_policy_action_.has_value()) {
    command.source_request_id = last_policy_action_->request_id;
    command.source_generation_id = last_policy_action_->generation_id;
    command.source_observation_step = last_policy_action_->source_observation_step;
    command.source_target_step = last_policy_action_->source_target_step;
  }
  last_command_ = command.command;
  return command;
}

void ExecutorStateMachine::close_hold_locked(const std::uint64_t now_steady_time_ns)
{
  if (!hold_started_steady_time_ns_.has_value()) {
    return;
  }
  if (now_steady_time_ns >= hold_started_steady_time_ns_.value()) {
    total_hold_duration_ns_ += now_steady_time_ns - hold_started_steady_time_ns_.value();
  }
  hold_started_steady_time_ns_.reset();
}

std::optional<CommandRecord> ExecutorStateMachine::command_for_step(
  const std::string & episode_id, const std::uint64_t actual_target_step,
  const ClockStamp & stamp)
{
  std::scoped_lock lock(mutex_);
  if (episode_id != episode_id_ || !episode_active_) {
    RuntimeEventRecord event;
    event.stamp = stamp;
    event.event_type = "command_rejected";
    event.reason = episode_id != episode_id_ ? "previous_episode" : "episode_not_active";
    event.episode_id = episode_id;
    event.actual_target_step = actual_target_step;
    push_event_locked(std::move(event));
    return std::nullopt;
  }
  if (latest_executed_target_step_.has_value() &&
    actual_target_step <= latest_executed_target_step_.value())
  {
    RuntimeEventRecord event;
    event.stamp = stamp;
    event.event_type = "command_rejected";
    event.reason = "target_already_executed";
    event.actual_target_step = actual_target_step;
    event.queue_length_before = queue_.size();
    event.queue_length_after = queue_.size();
    push_event_locked(std::move(event));
    return std::nullopt;
  }
  if (latest_executed_target_step_.has_value() &&
    actual_target_step > latest_executed_target_step_.value() + 1U)
  {
    deadline_misses_ += actual_target_step - latest_executed_target_step_.value() - 1U;
  }

  const auto queue_before = queue_.size();
  if (strategy_ != Strategy::kNaiveAsync) {
    while (!queue_.empty() && queue_.front().source_target_step < actual_target_step) {
      RuntimeEventRecord expired_event;
      expired_event.stamp = stamp;
      expired_event.event_type = "action_discarded";
      expired_event.reason = "expired_before_execution";
      expired_event.request_id = queue_.front().request_id;
      expired_event.generation_id = queue_.front().generation_id;
      expired_event.source_observation_step = queue_.front().source_observation_step;
      expired_event.source_target_step = queue_.front().source_target_step;
      expired_event.actual_target_step = actual_target_step;
      expired_event.queue_length_before = queue_.size();
      queue_.pop_front();
      expired_event.queue_length_after = queue_.size();
      expired_event.action_count = 1U;
      ++expired_actions_removed_;
      push_event_locked(std::move(expired_event));
    }
  }

  std::optional<QueuedAction> selected;
  if (strategy_ == Strategy::kNaiveAsync) {
    if (!queue_.empty()) {
      selected = std::move(queue_.front());
      queue_.pop_front();
    }
  } else if (!queue_.empty() && queue_.front().source_target_step == actual_target_step) {
    selected = std::move(queue_.front());
    queue_.pop_front();
  }

  CommandRecord command;
  if (selected.has_value()) {
    close_hold_locked(stamp.steady_time_ns);
    command.stamp = stamp;
    command.episode_id = episode_id_;
    command.actual_target_step = actual_target_step;
    command.source_request_id = selected->request_id;
    command.source_generation_id = selected->generation_id;
    command.source_observation_step = selected->source_observation_step;
    command.source_target_step = selected->source_target_step;
    command.command = selected->command;
    command.hold = false;
    command.reason = strategy_ == Strategy::kNaiveAsync ?
      "arrival_order_execution" : "target_step_match";
    last_policy_action_ = selected;
    last_command_ = command.command;
    current_action_age_steps_ = actual_target_step >= selected->source_observation_step ?
      actual_target_step - selected->source_observation_step : 0U;
    executed_source_generation_id_ = selected->generation_id;
  } else {
    std::string reason = "queue_empty";
    if (strategy_ == Strategy::kSyncHold && sync_request_in_flight_.has_value()) {
      reason = "waiting_for_sync_response";
    } else if (!queue_.empty()) {
      reason = "future_target_not_ready";
    }
    command = make_hold_locked(actual_target_step, stamp, reason);
  }

  latest_executed_target_step_ = actual_target_step;
  RuntimeEventRecord event;
  event.stamp = stamp;
  event.event_type = command.hold ? "hold_executed" : "action_executed";
  event.reason = command.reason;
  event.request_id = command.source_request_id;
  event.generation_id = command.source_generation_id;
  event.source_observation_step = command.source_observation_step;
  event.source_target_step = command.source_target_step;
  event.actual_target_step = actual_target_step;
  event.queue_length_before = queue_before;
  event.queue_length_after = queue_.size();
  event.action_count = 1U;
  push_event_locked(std::move(event));
  return command;
}

Diagnostics ExecutorStateMachine::diagnostics(const std::uint64_t now_steady_time_ns) const
{
  std::scoped_lock lock(mutex_);
  Diagnostics diagnostics;
  diagnostics.strategy = strategy_;
  diagnostics.episode_id = episode_id_;
  diagnostics.episode_active = episode_active_;
  diagnostics.episode_terminated = episode_terminated_;
  diagnostics.episode_success = episode_success_;
  diagnostics.latest_observation_step = latest_observation_step_;
  diagnostics.has_executed_target_step = latest_executed_target_step_.has_value();
  diagnostics.latest_executed_target_step = latest_executed_target_step_.value_or(0U);
  diagnostics.active_generation_id = active_generation_id_;
  diagnostics.queue_length = queue_.size();
  diagnostics.accepted_chunks = accepted_chunks_;
  diagnostics.rejected_chunks = rejected_chunks_;
  diagnostics.rejected_previous_episode_chunks = rejected_previous_episode_chunks_;
  diagnostics.rejected_stale_generation_chunks = rejected_stale_generation_chunks_;
  diagnostics.rejected_unknown_request_chunks = rejected_unknown_request_chunks_;
  diagnostics.duplicate_responses = duplicate_responses_;
  diagnostics.expired_actions_removed = expired_actions_removed_;
  diagnostics.duplicate_actions_removed = duplicate_actions_removed_;
  diagnostics.queue_rebuilds = queue_rebuilds_;
  diagnostics.deadline_misses = deadline_misses_;
  diagnostics.hold_steps = hold_steps_;
  diagnostics.total_hold_duration_ns = total_hold_duration_ns_;
  if (hold_started_steady_time_ns_.has_value() &&
    now_steady_time_ns >= hold_started_steady_time_ns_.value())
  {
    diagnostics.current_hold_duration_ns =
      now_steady_time_ns - hold_started_steady_time_ns_.value();
  }
  diagnostics.current_action_age_steps = current_action_age_steps_;
  diagnostics.executed_source_generation_id = executed_source_generation_id_;
  return diagnostics;
}

std::vector<RuntimeEventRecord> ExecutorStateMachine::drain_events()
{
  std::scoped_lock lock(mutex_);
  auto result = std::move(events_);
  events_.clear();
  return result;
}

void ExecutorStateMachine::push_event_locked(RuntimeEventRecord event)
{
  event.strategy = strategy_;
  if (event.episode_id.empty()) {
    event.episode_id = episode_id_;
  }
  event.active_generation_id = active_generation_id_;
  events_.push_back(std::move(event));
}

void ExecutorStateMachine::reset_counters_locked()
{
  accepted_chunks_ = 0U;
  rejected_chunks_ = 0U;
  rejected_previous_episode_chunks_ = 0U;
  rejected_stale_generation_chunks_ = 0U;
  rejected_unknown_request_chunks_ = 0U;
  duplicate_responses_ = 0U;
  expired_actions_removed_ = 0U;
  duplicate_actions_removed_ = 0U;
  queue_rebuilds_ = 0U;
  deadline_misses_ = 0U;
  hold_steps_ = 0U;
  total_hold_duration_ns_ = 0U;
  current_action_age_steps_ = 0U;
  executed_source_generation_id_ = 0U;
}

}  // namespace action_stream_executor
