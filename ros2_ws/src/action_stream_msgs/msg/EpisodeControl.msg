# REQUEST messages drive lifecycle transitions. STATUS messages are ACK/state
# publications on the shared topic and must never be treated as new requests.
uint8 REQUEST=0
uint8 STATUS=1

uint8 START=0
uint8 RESET=1
uint8 TERMINATE=2

uint8 message_kind
uint8 command
builtin_interfaces/Time sim_stamp
uint64 steady_time_ns
uint64 wall_time_ns
string episode_id
uint64 generation_id
bool active
bool terminated
bool success
