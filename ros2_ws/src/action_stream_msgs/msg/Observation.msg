# Simulator state captured after observation_step completed. The next command
# produced from this observation targets observation_step + 1.
builtin_interfaces/Time sim_stamp
uint64 steady_time_ns
uint64 wall_time_ns
string episode_id
uint64 observation_step
uint64 generation_id
string task_id
float64[] robot_state
float64[] task_state
bool terminated
