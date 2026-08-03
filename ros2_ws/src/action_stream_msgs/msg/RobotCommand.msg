# actual_target_step is the simulator step receiving this command.
# source_target_step preserves the policy label even for naive_async.
builtin_interfaces/Time sim_stamp
uint64 steady_time_ns
uint64 wall_time_ns
string episode_id
uint64 actual_target_step
uint64 source_request_id
uint64 source_generation_id
uint64 source_observation_step
uint64 source_target_step
float64[] command
bool hold
string reason
