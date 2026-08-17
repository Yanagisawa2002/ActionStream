#include "action_stream_executor/executor_state_machine.hpp"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace action_stream_executor
{
namespace
{

constexpr std::size_t kActionDimension = 7U;

[[nodiscard]] std::vector<std::string> split(const std::string & value, const char delimiter)
{
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    result.push_back(item);
  }
  if (!value.empty() && value.back() == delimiter) {
    result.emplace_back();
  }
  return result;
}

[[nodiscard]] std::uint64_t unsigned_value(const std::string & value)
{
  std::size_t consumed = 0U;
  const auto result = std::stoull(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid unsigned integer: " + value);
  }
  return result;
}

[[nodiscard]] bool boolean_value(const std::string & value)
{
  if (value == "1" || value == "true") {
    return true;
  }
  if (value == "0" || value == "false") {
    return false;
  }
  throw std::invalid_argument("invalid boolean: " + value);
}

[[nodiscard]] ClockStamp stamp(const std::uint64_t tick)
{
  return ClockStamp{
    static_cast<std::int64_t>(tick * 50'000'000U), 1'000U + tick, 10'000U + tick};
}

[[nodiscard]] std::vector<TargetActionRecord> actions(const std::string & specification)
{
  std::vector<TargetActionRecord> result;
  if (specification.empty()) {
    return result;
  }
  for (const auto & item : split(specification, ',')) {
    const auto parts = split(item, ':');
    if (parts.size() != 2U) {
      throw std::invalid_argument("action must be target:token: " + item);
    }
    const auto target = unsigned_value(parts[0]);
    const auto token = std::stod(parts[1]);
    result.push_back(TargetActionRecord{target, std::vector<double>(kActionDimension, token)});
  }
  return result;
}

[[nodiscard]] std::string queue_field(const StateSnapshot & snapshot)
{
  std::ostringstream output;
  output << std::setprecision(17);
  bool first = true;
  for (const auto & action : snapshot.queue) {
    if (!first) {
      output << ';';
    }
    first = false;
    output << action.request_id << ',' << action.generation_id << ','
           << action.source_observation_step << ',' << action.source_target_step << ','
           << (action.command.empty() ? 0.0 : action.command.front());
  }
  return output.str();
}

[[nodiscard]] std::string events_field(const std::vector<RuntimeEventRecord> & events)
{
  std::ostringstream output;
  bool first = true;
  for (const auto & event : events) {
    if (!first) {
      output << ';';
    }
    first = false;
    output << event.event_type << '~' << event.reason << '~' << event.request_id << '~'
           << event.generation_id << '~' << event.source_observation_step << '~'
           << event.source_target_step << '~' << event.actual_target_step << '~'
           << event.active_generation_id << '~' << event.queue_length_before << '~'
           << event.queue_length_after << '~' << event.action_count;
  }
  return output.str();
}

void print_header()
{
  std::cout
    << "event_id|operation|accepted|reason|actions_expired|duplicates_removed|"
       "decision_queue_length|command_present|command_hold|command_reason|command_request|"
       "command_generation|command_source_observation|command_source_target|command_actual_target|"
       "command_token|episode_active|episode_terminated|latest_observation|active_generation|"
       "latest_source_present|latest_source_observation|latest_source_request|latest_executed_present|"
       "latest_executed_target|sync_inflight_present|sync_inflight_request|queue|events\n";
}

void print_result(
  const std::string & event_id, const std::string & operation,
  const std::optional<Decision> & decision, const std::optional<CommandRecord> & command,
  const StateSnapshot & snapshot, const std::vector<RuntimeEventRecord> & events)
{
  const auto accepted = decision.has_value() ? decision->accepted : true;
  const auto reason = decision.has_value() ? decision->reason : "no_state_decision";
  std::cout << std::setprecision(17) << event_id << '|' << operation << '|'
            << (accepted ? 1 : 0) << '|' << reason << '|'
            << (decision.has_value() ? decision->actions_expired : 0U) << '|'
            << (decision.has_value() ? decision->duplicates_removed : 0U) << '|'
            << (decision.has_value() ? decision->queue_length : snapshot.queue.size()) << '|'
            << (command.has_value() ? 1 : 0) << '|'
            << (command.has_value() && command->hold ? 1 : 0) << '|'
            << (command.has_value() ? command->reason : "") << '|'
            << (command.has_value() ? command->source_request_id : 0U) << '|'
            << (command.has_value() ? command->source_generation_id : 0U) << '|'
            << (command.has_value() ? command->source_observation_step : 0U) << '|'
            << (command.has_value() ? command->source_target_step : 0U) << '|'
            << (command.has_value() ? command->actual_target_step : 0U) << '|'
            << (command.has_value() && !command->command.empty() ? command->command.front() : 0.0)
            << '|' << (snapshot.episode_active ? 1 : 0) << '|'
            << (snapshot.episode_terminated ? 1 : 0) << '|'
            << snapshot.latest_observation_step << '|' << snapshot.active_generation_id << '|'
            << (snapshot.latest_accepted_plan_key.has_value() ? 1 : 0) << '|'
            << (snapshot.latest_accepted_plan_key.has_value() ?
              snapshot.latest_accepted_plan_key->first : 0U) << '|'
            << (snapshot.latest_accepted_plan_key.has_value() ?
              snapshot.latest_accepted_plan_key->second : 0U) << '|'
            << (snapshot.latest_executed_target_step.has_value() ? 1 : 0) << '|'
            << snapshot.latest_executed_target_step.value_or(0U) << '|'
            << (snapshot.sync_request_in_flight.has_value() ? 1 : 0) << '|'
            << snapshot.sync_request_in_flight.value_or(0U) << '|'
            << queue_field(snapshot) << '|' << events_field(events) << '\n';
}

}  // namespace
}  // namespace action_stream_executor

int main(int argc, char ** argv)
{
  using namespace action_stream_executor;
  try {
    if (argc < 2 || argc > 3) {
      throw std::invalid_argument(
              "usage: action_stream_executor_replay_probe STRATEGY [SYNC_PERIODIC_REPLAN]");
    }
    const auto strategy = parse_strategy(argv[1]);
    const auto sync_periodic_replan = argc == 3 ? boolean_value(argv[2]) : false;
    ExecutorStateMachine machine(
      strategy, kActionDimension,
      {0.45, 0.0, 0.35, 3.141592653589793, 0.0, 0.0, 1.0},
      sync_periodic_replan);

    print_header();
    std::string line;
    while (std::getline(std::cin, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.size() >= 3U && static_cast<unsigned char>(line[0]) == 0xEFU &&
        static_cast<unsigned char>(line[1]) == 0xBBU &&
        static_cast<unsigned char>(line[2]) == 0xBFU)
      {
        line.erase(0U, 3U);
      }
      if (line.empty() || line.front() == '#') {
        continue;
      }
      const auto fields = split(line, '|');
      if (fields.size() < 2U) {
        throw std::invalid_argument("trace line needs event ID and operation");
      }
      const auto & event_id = fields[0];
      const auto & operation = fields[1];
      std::optional<Decision> decision;
      std::optional<CommandRecord> command;

      if (operation == "START" || operation == "RESET") {
        if (fields.size() != 6U) {
          throw std::invalid_argument(operation + " expects 4 arguments");
        }
        const auto event_stamp = stamp(unsigned_value(fields[5]));
        if (operation == "START") {
          decision = machine.start_episode(
            fields[2], unsigned_value(fields[3]), unsigned_value(fields[4]), event_stamp);
        } else {
          decision = machine.reset_episode(
            fields[2], unsigned_value(fields[3]), unsigned_value(fields[4]), event_stamp);
        }
      } else if (operation == "OBSERVE") {
        if (fields.size() != 7U) {
          throw std::invalid_argument("OBSERVE expects 5 arguments");
        }
        decision = machine.observe(
          ObservationRecord{
            stamp(unsigned_value(fields[6])), fields[2], unsigned_value(fields[3]),
            boolean_value(fields[5]), unsigned_value(fields[4])});
      } else if (operation == "REQUEST") {
        if (fields.size() != 8U) {
          throw std::invalid_argument("REQUEST expects 6 arguments");
        }
        const auto source_step = unsigned_value(fields[5]);
        const auto source_stamp = stamp(source_step);
        decision = machine.register_request(
          RequestRecord{
            stamp(unsigned_value(fields[7])), fields[2], unsigned_value(fields[3]),
            unsigned_value(fields[4]), source_step, source_stamp.sim_time_ns,
            source_stamp.steady_time_ns, static_cast<std::uint32_t>(unsigned_value(fields[6]))});
      } else if (operation == "CHUNK") {
        if (fields.size() != 8U) {
          throw std::invalid_argument("CHUNK expects 6 arguments");
        }
        const auto tick = unsigned_value(fields[6]);
        const auto source_step = unsigned_value(fields[5]);
        const auto source_stamp = stamp(source_step);
        ChunkRecord chunk;
        chunk.stamp = stamp(tick);
        chunk.episode_id = fields[2];
        chunk.request_id = unsigned_value(fields[3]);
        chunk.generation_id = unsigned_value(fields[4]);
        chunk.source_observation_step = source_step;
        chunk.source_sim_time_ns = source_stamp.sim_time_ns;
        chunk.source_observation_steady_time_ns = source_stamp.steady_time_ns;
        chunk.action_dimension = static_cast<std::uint32_t>(kActionDimension);
        chunk.actions = actions(fields[7]);
        chunk.inference_complete_steady_time_ns = chunk.stamp.steady_time_ns - 2U;
        chunk.response_publish_steady_time_ns = chunk.stamp.steady_time_ns - 1U;
        decision = machine.ingest_chunk(chunk);
      } else if (operation == "COMMAND") {
        if (fields.size() != 5U) {
          throw std::invalid_argument("COMMAND expects 3 arguments");
        }
        command = machine.command_for_step(
          fields[2], unsigned_value(fields[3]), stamp(unsigned_value(fields[4])));
      } else if (operation == "TERMINATE") {
        if (fields.size() != 5U) {
          throw std::invalid_argument("TERMINATE expects 3 arguments");
        }
        decision = machine.terminate_episode(
          fields[2], boolean_value(fields[3]), stamp(unsigned_value(fields[4])));
      } else if (operation == "DROP") {
        if (fields.size() != 3U) {
          throw std::invalid_argument("DROP expects one request label");
        }
      } else {
        throw std::invalid_argument("unknown trace operation: " + operation);
      }

      const auto snapshot = machine.snapshot();
      print_result(event_id, operation, decision, command, snapshot, machine.drain_events());
    }
    return EXIT_SUCCESS;
  } catch (const std::exception & error) {
    std::cerr << "differential replay probe failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
