FROM ros:jazzy-ros-base@sha256:6513503d0b10e919fbe8134981d4f9d19b5c1f9b045b87a9fe3b0b9e03e7c2a9

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        python3-colcon-common-extensions \
        python3-matplotlib \
        python3-numpy \
        python3-pytest \
        python3-scipy \
        python3-yaml \
        ros-jazzy-ament-cmake-gtest \
        ros-jazzy-ros2launch \
        ros-jazzy-rosgraph-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ros2_ws

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
