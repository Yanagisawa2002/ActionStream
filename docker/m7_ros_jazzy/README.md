# M7 ROS 2 Jazzy build image

This image is pinned to the exact official `ros:jazzy-ros-base` digest recorded
by the M7 starting-state audit. It contains only the compiler, ROS build/test
tools, and CPU analysis dependencies needed by the deterministic ROS 2 fallback
benchmark. It does not contain Isaac Sim and must never be reported as an Isaac
Sim environment.

Build from the repository root:

```powershell
docker build -t actionstream-m7-ros:jazzy docker/m7_ros_jazzy
```
