# ROS2-Jazzy Podman Compose Commands

@default:
    just --list

# Start all nodes using Podman Compose in detached mode
up:
    podman compose up -d

# Stop and remove all containers and the resources started by Podman Compose
down:
    podman compose down

# View logs for a specific service
log service="node0":
    podman compose logs -f {{ service }}

# Run all nodes
[parallel]
run: run-talker run-listener

# Build all packages
[parallel]
build: build-talker build-listener

# Run node0
[group("Run Individual")]
run-talker:
    podman compose exec -it node0 bash -lc "source /NNUF/install/setup.bash && ros2 run talker_pkg talker"

# Run node1
[group("Run Individual")]
run-listener:
    podman compose exec -it node1 bash -lc "source /NNUF/install/setup.bash && ros2 run listener_pkg listener"

# Build 'node0 package'
[group("Build Individual")]
build-talker:
    podman compose exec -it node0 bash -lc "colcon build --packages-select talker_pkg"

# Build 'node1 package'
[group("Build Individual")]
build-listener:
    podman compose exec -it node1 bash -lc "colcon build --packages-select listener_pkg"
