#include "action_stream_executor/executor_state_machine.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace action_stream_executor
{
namespace
{

constexpr std::size_t kActionDimension = 2U;

[[nodiscard]] ClockStamp stamp(const std::uint64_t tick)
{
  return ClockStamp{
    static_cast<std::int64_t>(tick * 50'000'000U), 1'000U + tick, 10'000U + tick};
}

[[nodiscard]] RequestRecord request(
  const std::uint64_t request_id, const std::uint64_t generation_id,
  const std::uint64_t source_step, const std::uint32_t horizon,
  const std::string & episode_id = "episode")
{
  RequestRecord result;
  result.stamp = stamp(source_step);
  result.episode_id = episode_id;
  result.request_id = request_id;
  result.generation_id = generation_id;
  result.source_observation_step = source_step;
  result.source_sim_time_ns = stamp(source_step).sim_time_ns;
  result.source_observation_steady_time_ns = stamp(source_step).steady_time_ns;
  result.expected_horizon = horizon;
  return result;
}

[[nodiscard]] ChunkRecord chunk(
  const RequestRecord & source_request, std::vector<std::uint64_t> targets,
  const double value_offset = 0.0)
{
  ChunkRecord result;
  result.stamp = stamp(source_request.source_observation_step + 20U);
  result.episode_id = source_request.episode_id;
  result.request_id = source_request.request_id;
  result.generation_id = source_request.generation_id;
  result.source_observation_step = source_request.source_observation_step;
  result.source_sim_time_ns = source_request.source_sim_time_ns;
  result.source_observation_steady_time_ns = source_request.source_observation_steady_time_ns;
  result.action_dimension = static_cast<std::uint32_t>(kActionDimension);
  result.inference_complete_steady_time_ns = result.stamp.steady_time_ns - 2U;
  result.response_publish_steady_time_ns = result.stamp.steady_time_ns - 1U;
  for (const auto target : targets) {
    result.actions.push_back(
      TargetActionRecord{
        target, {value_offset + static_cast<double>(target), value_offset + 0.5}});
  }
  return result;
}

void start(
  ExecutorStateMachine & machine, const std::uint64_t observation_step = 0U,
  const std::uint64_t generation = 0U, const std::string & episode_id = "episode")
{
  ASSERT_TRUE(
    machine.start_episode(episode_id, generation, observation_step, stamp(observation_step)).accepted);
  static_cast<void>(machine.drain_events());
}

void observe(
  ExecutorStateMachine & machine, const std::uint64_t observation_step,
  const std::string & episode_id = "episode")
{
  ASSERT_TRUE(
    machine.observe(
      ObservationRecord{stamp(observation_step), episode_id, observation_step, false}).accepted);
}

TEST(StrategyTest, ParsesOnlyDeclaredStrategies)
{
  EXPECT_EQ(parse_strategy("sync_hold"), Strategy::kSyncHold);
  EXPECT_EQ(parse_strategy("naive_async"), Strategy::kNaiveAsync);
  EXPECT_EQ(parse_strategy("aligned_async"), Strategy::kAlignedAsync);
  EXPECT_THROW(static_cast<void>(parse_strategy("latest")), std::invalid_argument);
}

TEST(EpisodeControlFilterTest, ProcessesRequestsAndIgnoresStatusOrUnknownKinds)
{
  EXPECT_TRUE(should_process_episode_control(kEpisodeControlRequest));
  EXPECT_FALSE(should_process_episode_control(kEpisodeControlStatus));
  EXPECT_FALSE(should_process_episode_control(255U));
}

TEST(SafeHoldValidationTest, RequiresFiniteNonzeroCanonicalSevenDimensionalCommand)
{
  const std::vector<double> canonical{
    0.45, 0.0, 0.35, 3.141592653589793, 0.0, 0.0, 1.0};
  EXPECT_NO_THROW(validate_safe_hold_command(canonical, kCanonicalActionDimension));
  EXPECT_NO_THROW(
    ExecutorStateMachine(Strategy::kAlignedAsync, kCanonicalActionDimension, canonical));
  EXPECT_THROW(
    ExecutorStateMachine(
      Strategy::kAlignedAsync, kCanonicalActionDimension,
      std::vector<double>(kCanonicalActionDimension, 0.0)),
    std::invalid_argument);
  EXPECT_THROW(
    validate_safe_hold_command({}, kCanonicalActionDimension), std::invalid_argument);
  EXPECT_THROW(
    validate_safe_hold_command({0.45, 0.0}, kCanonicalActionDimension),
    std::invalid_argument);
  auto nonfinite = canonical;
  nonfinite[2] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(
    validate_safe_hold_command(nonfinite, kCanonicalActionDimension),
    std::invalid_argument);
}

TEST(TimeConversionTest, NormalizesNegativeAndPositiveSimulationTimes)
{
  EXPECT_EQ(combine_time_ns(2, 3U), 2'000'000'003LL);
  const auto negative = split_time_ns(-1);
  EXPECT_EQ(negative.seconds, -1);
  EXPECT_EQ(negative.nanoseconds, 999'999'999U);
  EXPECT_EQ(combine_time_ns(negative.seconds, negative.nanoseconds), -1);
  EXPECT_THROW(static_cast<void>(combine_time_ns(0, 1'000'000'000U)), std::invalid_argument);
}

TEST(AlignedExecutorTest, EnforcesOPlusOneThroughOPlusHLabels)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 10U, 3U);
  const auto valid_request = request(1U, 3U, 10U, 3U);
  ASSERT_TRUE(machine.register_request(valid_request).accepted);
  const auto accepted = machine.ingest_chunk(chunk(valid_request, {11U, 12U, 13U}));
  ASSERT_TRUE(accepted.accepted);
  EXPECT_EQ(accepted.queue_length, 3U);

  const auto command = machine.command_for_step("episode", 11U, stamp(11U));
  ASSERT_TRUE(command.has_value());
  EXPECT_FALSE(command->hold);
  EXPECT_EQ(command->source_target_step, 11U);

  const auto malformed_request = request(2U, 3U, 10U, 3U);
  ASSERT_TRUE(machine.register_request(malformed_request).accepted);
  const auto rejected = machine.ingest_chunk(chunk(malformed_request, {10U, 11U, 12U}));
  EXPECT_FALSE(rejected.accepted);
  EXPECT_EQ(rejected.reason, "target_outside_horizon");
}

TEST(AlignedExecutorTest, RemovesArrivalAgePrefixUsingQueueInsertionStep)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 10U, 1U);
  const auto source_request = request(1U, 1U, 10U, 4U);
  ASSERT_TRUE(machine.register_request(source_request).accepted);
  observe(machine, 12U);

  const auto decision = machine.ingest_chunk(chunk(source_request, {11U, 12U, 13U, 14U}));
  ASSERT_TRUE(decision.accepted);
  EXPECT_EQ(decision.actions_expired, 2U);
  EXPECT_EQ(decision.queue_length, 2U);
  const auto command = machine.command_for_step("episode", 13U, stamp(13U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_target_step, 13U);
  EXPECT_EQ(machine.diagnostics(stamp(13U).steady_time_ns).expired_actions_removed, 2U);
}

TEST(AlignedExecutorTest, RejectsCompletelyExpiredChunkWithoutDestroyingQueue)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  const auto first = request(1U, 1U, 0U, 4U);
  ASSERT_TRUE(machine.register_request(first).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(first, {1U, 2U, 3U, 4U})).accepted);

  const auto expired = request(2U, 1U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(expired).accepted);
  observe(machine, 2U);
  const auto decision = machine.ingest_chunk(chunk(expired, {1U, 2U}));
  EXPECT_FALSE(decision.accepted);
  EXPECT_EQ(decision.reason, "fully_expired");
  EXPECT_EQ(decision.actions_expired, 2U);
  EXPECT_EQ(machine.diagnostics(stamp(2U).steady_time_ns).queue_length, 4U);
}

TEST(AlignedExecutorTest, RemovesDuplicateTargetsDeterministicallyFirstWins)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  const auto source_request = request(1U, 1U, 0U, 3U);
  ASSERT_TRUE(machine.register_request(source_request).accepted);
  auto duplicate_chunk = chunk(source_request, {1U, 1U, 2U});
  duplicate_chunk.actions[0].command = {10.0, 10.0};
  duplicate_chunk.actions[1].command = {99.0, 99.0};
  const auto decision = machine.ingest_chunk(duplicate_chunk);
  ASSERT_TRUE(decision.accepted);
  EXPECT_EQ(decision.duplicates_removed, 1U);
  EXPECT_EQ(decision.queue_length, 2U);
  const auto command = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->command, (std::vector<double>{10.0, 10.0}));
}

TEST(AlignedExecutorTest, MalformedReplacementLeavesOldQueueAtomicallyIntact)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  const auto first = request(1U, 1U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(first).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(first, {1U, 2U}, 10.0)).accepted);

  const auto second = request(2U, 1U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(second).accepted);
  auto malformed = chunk(second, {1U, 2U}, 20.0);
  malformed.actions[1].command = {1.0};
  const auto decision = machine.ingest_chunk(malformed);
  EXPECT_FALSE(decision.accepted);
  EXPECT_EQ(decision.reason, "invalid_command");
  EXPECT_EQ(machine.diagnostics(stamp(0U).steady_time_ns).queue_length, 2U);
  const auto command = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(command.has_value());
  EXPECT_DOUBLE_EQ(command->command.front(), 11.0);
}

TEST(AlignedExecutorTest, ValidNewerChunkAtomicallyReplacesQueue)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  const auto first = request(1U, 1U, 0U, 3U);
  ASSERT_TRUE(machine.register_request(first).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(first, {1U, 2U, 3U}, 10.0)).accepted);
  const auto second = request(2U, 1U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(second).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(second, {1U, 2U}, 20.0)).accepted);
  EXPECT_EQ(machine.diagnostics(stamp(0U).steady_time_ns).queue_length, 2U);
  const auto command = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_request_id, 2U);
  EXPECT_DOUBLE_EQ(command->command.front(), 21.0);
}

TEST(GenerationTest, GenerationFiveResponseWinsBeforeGenerationFour)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 4U);
  const auto generation_four = request(4U, 4U, 0U, 2U);
  const auto generation_five = request(5U, 5U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(generation_four).accepted);
  ASSERT_TRUE(machine.register_request(generation_five).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(generation_five, {1U, 2U}, 50.0)).accepted);
  const auto late = machine.ingest_chunk(chunk(generation_four, {1U, 2U}, 40.0));
  EXPECT_FALSE(late.accepted);
  EXPECT_EQ(late.reason, "stale_generation");
  const auto command = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_generation_id, 5U);
  EXPECT_EQ(machine.diagnostics(stamp(1U).steady_time_ns).rejected_stale_generation_chunks, 1U);
}

TEST(GenerationTest, DroppedRequestDoesNotBlockNewerSuccessfulGeneration)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  ASSERT_TRUE(machine.register_request(request(1U, 1U, 0U, 2U)).accepted);
  const auto newer = request(2U, 2U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(newer).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(newer, {1U, 2U})).accepted);
  const auto command = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_generation_id, 2U);
}

TEST(FreshnessTest, FresherSameGenerationResponseCannotBeOverwrittenByOlderSource)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 11U, 7U);
  const auto older = request(100U, 7U, 10U, 3U);
  const auto fresher = request(101U, 7U, 11U, 3U);
  ASSERT_TRUE(machine.register_request(older).accepted);
  ASSERT_TRUE(machine.register_request(fresher).accepted);

  ASSERT_TRUE(machine.ingest_chunk(chunk(fresher, {12U, 13U, 14U}, 100.0)).accepted);
  const auto late_older = machine.ingest_chunk(chunk(older, {11U, 12U, 13U}, 10.0));
  EXPECT_FALSE(late_older.accepted);
  EXPECT_EQ(late_older.reason, "stale_plan_freshness");
  EXPECT_EQ(late_older.queue_length, 3U);

  const auto command = machine.command_for_step("episode", 12U, stamp(12U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_request_id, 101U);
  EXPECT_EQ(command->source_observation_step, 11U);

  const auto events = machine.drain_events();
  const auto rejected = std::find_if(
    events.begin(), events.end(), [](const RuntimeEventRecord & event) {
      return event.event_type == "chunk_rejected" &&
             event.reason == "stale_plan_freshness";
    });
  ASSERT_NE(rejected, events.end());
  EXPECT_NE(rejected->detail.find("latest_source_step=11"), std::string::npos);
  EXPECT_NE(rejected->detail.find("latest_request_id=101"), std::string::npos);
}

TEST(FreshnessTest, RequestIdBreaksSameSourceTieDeterministically)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 5U, 2U);
  const auto lower_id = request(10U, 2U, 5U, 2U);
  const auto higher_id = request(11U, 2U, 5U, 2U);
  ASSERT_TRUE(machine.register_request(lower_id).accepted);
  ASSERT_TRUE(machine.register_request(higher_id).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(higher_id, {6U, 7U}, 20.0)).accepted);
  EXPECT_EQ(
    machine.ingest_chunk(chunk(lower_id, {6U, 7U}, 10.0)).reason,
    "stale_plan_freshness");

  ExecutorStateMachine forward_machine(
    Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(forward_machine, 5U, 2U);
  ASSERT_TRUE(forward_machine.register_request(lower_id).accepted);
  ASSERT_TRUE(forward_machine.register_request(higher_id).accepted);
  ASSERT_TRUE(forward_machine.ingest_chunk(chunk(lower_id, {6U, 7U}, 10.0)).accepted);
  ASSERT_TRUE(forward_machine.ingest_chunk(chunk(higher_id, {6U, 7U}, 20.0)).accepted);
  const auto command = forward_machine.command_for_step("episode", 6U, stamp(6U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_request_id, 11U);
}

TEST(FreshnessTest, EpisodeAndGenerationResetClearAcceptedPlanOrdering)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 5U, 1U);
  const auto high_key = request(100U, 1U, 5U, 2U);
  ASSERT_TRUE(machine.register_request(high_key).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(high_key, {6U, 7U})).accepted);

  const auto generation_reset = request(1U, 2U, 5U, 2U);
  ASSERT_TRUE(machine.register_request(generation_reset).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(generation_reset, {6U, 7U})).accepted);

  ASSERT_TRUE(machine.reset_episode("episode", 2U, 5U, stamp(5U)).accepted);
  const auto episode_reset = request(0U, 2U, 5U, 2U);
  ASSERT_TRUE(machine.register_request(episode_reset).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(episode_reset, {6U, 7U})).accepted);
  const auto command = machine.command_for_step("episode", 6U, stamp(6U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_request_id, 0U);
}

TEST(EpisodeLifecycleTest, ResetWhilePendingRejectsOldGeneration)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 4U);
  const auto pending = request(1U, 4U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(pending).accepted);
  ASSERT_TRUE(machine.reset_episode("episode", 5U, 0U, stamp(1U)).accepted);
  const auto decision = machine.ingest_chunk(chunk(pending, {1U, 2U}));
  EXPECT_FALSE(decision.accepted);
  EXPECT_EQ(decision.reason, "stale_generation");
}

TEST(EpisodeLifecycleTest, PreviousEpisodeAndPostTerminationResultsAreRejected)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U, "new_episode");
  const auto old_request = request(1U, 1U, 0U, 2U, "old_episode");
  EXPECT_EQ(
    machine.ingest_chunk(chunk(old_request, {1U, 2U})).reason,
    "previous_episode");

  const auto pending = request(2U, 1U, 0U, 2U, "new_episode");
  ASSERT_TRUE(machine.register_request(pending).accepted);
  ASSERT_TRUE(machine.terminate_episode("new_episode", true, stamp(1U)).accepted);
  const auto terminated = machine.ingest_chunk(chunk(pending, {1U, 2U}));
  EXPECT_FALSE(terminated.accepted);
  EXPECT_EQ(terminated.reason, "episode_terminated");
  EXPECT_TRUE(machine.diagnostics(stamp(1U).steady_time_ns).episode_success);
}

TEST(ResponseTest, DuplicateResponseIsRejectedWithoutQueueMutation)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  const auto source_request = request(1U, 1U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(source_request).accepted);
  const auto response = chunk(source_request, {1U, 2U});
  ASSERT_TRUE(machine.ingest_chunk(response).accepted);
  const auto duplicate = machine.ingest_chunk(response);
  EXPECT_FALSE(duplicate.accepted);
  EXPECT_EQ(duplicate.reason, "duplicate_response");
  const auto diagnostics = machine.diagnostics(stamp(0U).steady_time_ns);
  EXPECT_EQ(diagnostics.queue_length, 2U);
  EXPECT_EQ(diagnostics.duplicate_responses, 1U);
}

TEST(NaiveExecutorTest, AppendsChunksAndExecutesInResponseArrivalOrder)
{
  ExecutorStateMachine machine(Strategy::kNaiveAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 0U);
  const auto first_request = request(1U, 0U, 0U, 2U);
  const auto second_request = request(2U, 0U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(first_request).accepted);
  ASSERT_TRUE(machine.register_request(second_request).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(second_request, {1U, 2U}, 20.0)).accepted);
  ASSERT_TRUE(machine.ingest_chunk(chunk(first_request, {1U, 2U}, 10.0)).accepted);

  const auto command_one = machine.command_for_step("episode", 1U, stamp(1U));
  const auto command_two = machine.command_for_step("episode", 2U, stamp(2U));
  const auto command_three = machine.command_for_step("episode", 3U, stamp(3U));
  ASSERT_TRUE(command_one.has_value());
  ASSERT_TRUE(command_two.has_value());
  ASSERT_TRUE(command_three.has_value());
  EXPECT_EQ(command_one->source_request_id, 2U);
  EXPECT_EQ(command_two->source_request_id, 2U);
  EXPECT_EQ(command_three->source_request_id, 1U);
  EXPECT_EQ(command_three->source_target_step, 1U);
  EXPECT_EQ(command_three->actual_target_step, 3U);
}

TEST(SyncExecutorTest, HoldsSafelyUntilCorrespondingResponseThenExecutesValidSuffix)
{
  ExecutorStateMachine machine(Strategy::kSyncHold, kActionDimension, {9.0, 8.0});
  start(machine, 0U, 1U);
  const auto source_request = request(1U, 1U, 0U, 2U);
  ASSERT_TRUE(machine.register_request(source_request).accepted);
  const auto held = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(held.has_value());
  EXPECT_TRUE(held->hold);
  EXPECT_EQ(held->reason, "waiting_for_sync_response");
  EXPECT_EQ(held->command, (std::vector<double>{9.0, 8.0}));

  const auto decision = machine.ingest_chunk(chunk(source_request, {1U, 2U}));
  ASSERT_TRUE(decision.accepted);
  EXPECT_EQ(decision.actions_expired, 1U);
  const auto command = machine.command_for_step("episode", 2U, stamp(2U));
  ASSERT_TRUE(command.has_value());
  EXPECT_FALSE(command->hold);
  EXPECT_EQ(command->source_target_step, 2U);
  EXPECT_GT(machine.diagnostics(stamp(2U).steady_time_ns).total_hold_duration_ns, 0U);
}

TEST(CommandTest, EmptyQueueUsesConfiguredSafeHoldAndNeverExecutesOneStepTwice)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {3.0, 4.0});
  start(machine, 0U, 0U);
  const auto held = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(held.has_value());
  EXPECT_TRUE(held->hold);
  EXPECT_EQ(held->command, (std::vector<double>{3.0, 4.0}));
  EXPECT_FALSE(machine.command_for_step("episode", 1U, stamp(2U)).has_value());
  EXPECT_EQ(machine.diagnostics(stamp(2U).steady_time_ns).hold_steps, 1U);
}

TEST(BoundaryTest, RejectsTargetRangeOverflowAndZeroHorizon)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, std::numeric_limits<std::uint64_t>::max(), 1U);
  auto zero = request(
    1U, 1U, std::numeric_limits<std::uint64_t>::max(), 0U);
  EXPECT_EQ(machine.register_request(zero).reason, "zero_horizon");

  auto overflow = request(
    2U, 1U, std::numeric_limits<std::uint64_t>::max(), 1U);
  ASSERT_TRUE(machine.register_request(overflow).accepted);
  auto response = chunk(overflow, {std::numeric_limits<std::uint64_t>::max()});
  EXPECT_EQ(machine.ingest_chunk(response).reason, "target_step_overflow");
}

TEST(ConcurrencyTest, ConcurrentChunkCallbacksConvergeOnFreshestPlanAtomically)
{
  ExecutorStateMachine machine(Strategy::kAlignedAsync, kActionDimension, {0.0, 0.0});
  start(machine, 0U, 1U);
  constexpr std::uint64_t kRequestCount = 32U;
  std::vector<RequestRecord> requests;
  requests.reserve(kRequestCount);
  for (std::uint64_t index = 1U; index <= kRequestCount; ++index) {
    requests.push_back(request(index, 1U, 0U, 2U));
    ASSERT_TRUE(machine.register_request(requests.back()).accepted);
  }

  std::atomic<std::uint64_t> accepted{0U};
  std::vector<std::thread> workers;
  workers.reserve(kRequestCount);
  for (const auto & source_request : requests) {
    workers.emplace_back([&machine, &accepted, source_request]() {
        if (machine.ingest_chunk(
            chunk(source_request, {1U, 2U}, static_cast<double>(source_request.request_id))).accepted)
        {
          accepted.fetch_add(1U, std::memory_order_relaxed);
        }
      });
  }
  for (auto & worker : workers) {
    worker.join();
  }

  const auto accepted_count = accepted.load(std::memory_order_relaxed);
  EXPECT_GE(accepted_count, 1U);
  EXPECT_LE(accepted_count, kRequestCount);
  const auto diagnostics = machine.diagnostics(stamp(0U).steady_time_ns);
  EXPECT_EQ(diagnostics.accepted_chunks, accepted_count);
  EXPECT_EQ(diagnostics.queue_rebuilds, accepted_count);
  EXPECT_EQ(diagnostics.queue_length, 2U);
  EXPECT_EQ(diagnostics.rejected_chunks, kRequestCount - accepted_count);

  const auto command = machine.command_for_step("episode", 1U, stamp(1U));
  ASSERT_TRUE(command.has_value());
  EXPECT_EQ(command->source_request_id, kRequestCount);
  EXPECT_EQ(command->command, (std::vector<double>{33.0, 32.5}));
}

}  // namespace
}  // namespace action_stream_executor
