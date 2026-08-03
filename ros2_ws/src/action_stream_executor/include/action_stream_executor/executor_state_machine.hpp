#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace action_stream_executor
{

enum class Strategy
{
  kSyncHold,
  kNaiveAsync,
  kAlignedAsync,
};

[[nodiscard]] Strategy parse_strategy(std::string_view value);
[[nodiscard]] std::string to_string(Strategy strategy);

constexpr std::uint8_t kEpisodeControlRequest = 0U;
constexpr std::uint8_t kEpisodeControlStatus = 1U;
constexpr std::size_t kCanonicalActionDimension = 7U;

[[nodiscard]] bool should_process_episode_control(std::uint8_t message_kind) noexcept;
void validate_safe_hold_command(
  const std::vector<double> & command, std::size_t expected_dimension,
  bool reject_all_zero = true);

struct ClockStamp
{
  std::int64_t sim_time_ns{0};
  std::uint64_t steady_time_ns{0};
  std::uint64_t wall_time_ns{0};
};

struct TimeParts
{
  std::int32_t seconds{0};
  std::uint32_t nanoseconds{0};
};

[[nodiscard]] std::int64_t combine_time_ns(std::int32_t seconds, std::uint32_t nanoseconds);
[[nodiscard]] TimeParts split_time_ns(std::int64_t nanoseconds);

struct ObservationRecord
{
  ClockStamp stamp;
  std::string episode_id;
  std::uint64_t observation_step{0};
  bool terminated{false};
};

struct RequestRecord
{
  ClockStamp stamp;
  std::string episode_id;
  std::uint64_t request_id{0};
  std::uint64_t generation_id{0};
  std::uint64_t source_observation_step{0};
  std::int64_t source_sim_time_ns{0};
  std::uint64_t source_observation_steady_time_ns{0};
  std::uint32_t expected_horizon{0};
};

struct TargetActionRecord
{
  std::uint64_t target_step{0};
  std::vector<double> command;
};

struct ChunkRecord
{
  ClockStamp stamp;
  std::string episode_id;
  std::uint64_t request_id{0};
  std::uint64_t generation_id{0};
  std::uint64_t source_observation_step{0};
  std::int64_t source_sim_time_ns{0};
  std::uint64_t source_observation_steady_time_ns{0};
  std::uint32_t action_dimension{0};
  std::vector<TargetActionRecord> actions;
  std::uint64_t inference_complete_steady_time_ns{0};
  std::uint64_t response_publish_steady_time_ns{0};
};

struct CommandRecord
{
  ClockStamp stamp;
  std::string episode_id;
  std::uint64_t actual_target_step{0};
  std::uint64_t source_request_id{0};
  std::uint64_t source_generation_id{0};
  std::uint64_t source_observation_step{0};
  std::uint64_t source_target_step{0};
  std::vector<double> command;
  bool hold{false};
  std::string reason;
};

struct Decision
{
  bool accepted{false};
  std::string reason;
  std::size_t actions_inserted{0};
  std::size_t actions_expired{0};
  std::size_t duplicates_removed{0};
  std::size_t queue_length{0};
};

struct Diagnostics
{
  Strategy strategy{Strategy::kAlignedAsync};
  std::string episode_id;
  bool episode_active{false};
  bool episode_terminated{false};
  bool episode_success{false};
  std::uint64_t latest_observation_step{0};
  bool has_executed_target_step{false};
  std::uint64_t latest_executed_target_step{0};
  std::uint64_t active_generation_id{0};
  std::size_t queue_length{0};
  std::uint64_t accepted_chunks{0};
  std::uint64_t rejected_chunks{0};
  std::uint64_t rejected_previous_episode_chunks{0};
  std::uint64_t rejected_stale_generation_chunks{0};
  std::uint64_t rejected_unknown_request_chunks{0};
  std::uint64_t duplicate_responses{0};
  std::uint64_t expired_actions_removed{0};
  std::uint64_t duplicate_actions_removed{0};
  std::uint64_t queue_rebuilds{0};
  std::uint64_t deadline_misses{0};
  std::uint64_t hold_steps{0};
  std::uint64_t total_hold_duration_ns{0};
  std::uint64_t current_hold_duration_ns{0};
  std::uint64_t current_action_age_steps{0};
  std::uint64_t executed_source_generation_id{0};
};

struct RuntimeEventRecord
{
  ClockStamp stamp;
  std::string event_type;
  std::string reason;
  Strategy strategy{Strategy::kAlignedAsync};
  std::string episode_id;
  std::uint64_t request_id{0};
  std::uint64_t generation_id{0};
  std::uint64_t source_observation_step{0};
  std::uint64_t source_target_step{0};
  std::uint64_t actual_target_step{0};
  std::uint64_t active_generation_id{0};
  std::size_t queue_length_before{0};
  std::size_t queue_length_after{0};
  std::size_t action_count{0};
  std::string detail;
};

class ExecutorStateMachine
{
public:
  ExecutorStateMachine(
    Strategy strategy, std::size_t action_dimension, std::vector<double> safe_hold_command);

  ExecutorStateMachine(const ExecutorStateMachine &) = delete;
  ExecutorStateMachine & operator=(const ExecutorStateMachine &) = delete;

  [[nodiscard]] Decision start_episode(
    const std::string & episode_id, std::uint64_t initial_generation_id,
    std::uint64_t initial_observation_step, const ClockStamp & stamp);
  [[nodiscard]] Decision reset_episode(
    const std::string & episode_id, std::uint64_t initial_generation_id,
    std::uint64_t initial_observation_step, const ClockStamp & stamp);
  [[nodiscard]] Decision terminate_episode(
    const std::string & episode_id, bool success, const ClockStamp & stamp);
  [[nodiscard]] Decision observe(const ObservationRecord & observation);
  [[nodiscard]] Decision register_request(const RequestRecord & request);
  [[nodiscard]] Decision ingest_chunk(const ChunkRecord & chunk);
  [[nodiscard]] std::optional<CommandRecord> command_for_step(
    const std::string & episode_id, std::uint64_t actual_target_step, const ClockStamp & stamp);

  [[nodiscard]] Diagnostics diagnostics(std::uint64_t now_steady_time_ns) const;
  [[nodiscard]] std::vector<RuntimeEventRecord> drain_events();
  [[nodiscard]] Strategy strategy() const noexcept {return strategy_;}
  [[nodiscard]] std::size_t action_dimension() const noexcept {return action_dimension_;}

private:
  struct QueuedAction
  {
    std::uint64_t request_id{0};
    std::uint64_t generation_id{0};
    std::uint64_t source_observation_step{0};
    std::uint64_t source_target_step{0};
    std::vector<double> command;
  };

  [[nodiscard]] Decision begin_episode_locked(
    const std::string & event_type, const std::string & episode_id,
    std::uint64_t initial_generation_id, std::uint64_t initial_observation_step,
    const ClockStamp & stamp);
  [[nodiscard]] Decision reject_chunk_locked(
    const ChunkRecord & chunk, const std::string & reason, std::size_t queue_before,
    const std::string & detail = {});
  [[nodiscard]] bool validate_command_locked(const TargetActionRecord & action) const;
  [[nodiscard]] std::vector<QueuedAction> make_valid_actions_locked(
    const ChunkRecord & chunk, const RequestRecord & request, std::size_t * expired,
    std::size_t * duplicates, std::string * error) const;
  [[nodiscard]] CommandRecord make_hold_locked(
    std::uint64_t actual_target_step, const ClockStamp & stamp, const std::string & reason);
  void close_hold_locked(std::uint64_t now_steady_time_ns);
  void push_event_locked(RuntimeEventRecord event);
  void reset_counters_locked();

  const Strategy strategy_;
  const std::size_t action_dimension_;
  const std::vector<double> safe_hold_command_;

  mutable std::mutex mutex_;
  std::string episode_id_;
  bool episode_active_{false};
  bool episode_terminated_{false};
  bool episode_success_{false};
  std::uint64_t latest_observation_step_{0};
  std::optional<std::uint64_t> latest_executed_target_step_;
  std::uint64_t active_generation_id_{0};
  std::deque<QueuedAction> queue_;
  std::unordered_map<std::uint64_t, RequestRecord> requests_;
  std::unordered_set<std::uint64_t> completed_request_ids_;
  std::optional<std::pair<std::uint64_t, std::uint64_t>> latest_accepted_plan_key_;
  std::optional<std::uint64_t> sync_request_in_flight_;
  std::optional<QueuedAction> last_policy_action_;
  std::vector<double> last_command_;
  std::optional<std::uint64_t> hold_started_steady_time_ns_;
  std::vector<RuntimeEventRecord> events_;

  std::uint64_t accepted_chunks_{0};
  std::uint64_t rejected_chunks_{0};
  std::uint64_t rejected_previous_episode_chunks_{0};
  std::uint64_t rejected_stale_generation_chunks_{0};
  std::uint64_t rejected_unknown_request_chunks_{0};
  std::uint64_t duplicate_responses_{0};
  std::uint64_t expired_actions_removed_{0};
  std::uint64_t duplicate_actions_removed_{0};
  std::uint64_t queue_rebuilds_{0};
  std::uint64_t deadline_misses_{0};
  std::uint64_t hold_steps_{0};
  std::uint64_t total_hold_duration_ns_{0};
  std::uint64_t current_action_age_steps_{0};
  std::uint64_t executed_source_generation_id_{0};
};

}  // namespace action_stream_executor
