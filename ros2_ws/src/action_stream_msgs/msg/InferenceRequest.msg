# Immutable request provenance. source_observation_step is the last completed
# control step and a horizon-H response targets O+1 through O+H.
builtin_interfaces/Time sim_stamp
uint64 request_steady_time_ns
uint64 request_wall_time_ns
string episode_id
uint64 request_id
uint64 generation_id
uint64 source_observation_step
builtin_interfaces/Time source_sim_stamp
uint64 source_observation_steady_time_ns
uint32 expected_horizon
Observation observation
