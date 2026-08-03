# Structured reason codes are stable strings emitted by the pure state machine.
builtin_interfaces/Time sim_stamp
uint64 steady_time_ns
uint64 wall_time_ns
string event_type
string reason
string strategy
string episode_id
uint64 request_id
uint64 generation_id
uint64 source_observation_step
uint64 source_target_step
uint64 actual_target_step
uint64 active_generation_id
uint64 queue_length_before
uint64 queue_length_after
uint64 action_count
string detail
