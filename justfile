# ROS2-Jazzy Podman Compose Commands
host := "0.0.0.0"

@default:
    just --list

image:
    podman build -t nnuf:latest -f Dockerfile .

# Start all nodes using Podman Compose in detached mode
up: image
    podman compose up -d

# Stop and remove all containers and the resources started by Podman Compose
down:
    podman compose down

# View logs for a specific service
log service:
    podman compose logs -f {{ service }}

exec service:
    podman compose exec -it {{ service }} bash

# Open a shell in the mlops container
[group("Development")]
dev:
    podman compose exec -it mlops bash -lc "exec bash"

# Start marimo in mlops 'root directory: notebooks'
[group("Development")]
marimo url_or_file port="2718":
    podman compose exec -it mlops bash -lc "marimo edit --host {{ host }} --port {{ port }} --headless {{ url_or_file }}"

# Start Jupyter notebook in mlops 'root directory: notebook'
[group("Development")]
notebook folder_path="." port="8888":
    podman compose exec -it mlops bash -lc "jupyter notebook --ip {{ host }} --port {{ port }} --no-browser --allow-root {{ folder_path }}"

# Run all nodes (! Not Preffered)
[parallel]
run: run-talker run-listener run-seld

# Build all packages
[parallel]
build: build-talker build-listener build-seld

# Run node1
[group("Run Individual")]
run-talker:
    podman compose exec -it node1 bash -lc "source /NNUF/install/setup.bash && ros2 run talker_pkg talker"

# Run node2
[group("Run Individual")]
run-listener:
    podman compose exec -it node2 bash -lc "source /NNUF/install/setup.bash && ros2 run listener_pkg listener"

# Run node3
[group("Run Individual")]
run-seld:
    podman compose exec -it node3 bash -lc "source /NNUF/install/setup.bash && ros2 run seld_pkg seld"

[group("Run Individual")]
request-seld:
    podman compose exec -it node0 bash -lc "ros2 service call /run_inference rcl_interfaces/srv/SetParameters \"{parameters: [{name: 'audio_path', value: {type: 4, string_value: '/NNUF/seld_pkg/DCASE2025_SELD_dataset/stereo_eval/eval/sample06813.wav'}}, {name: 'video_path', value: {type: 4, string_value: '/NNUF/seld_pkg/DCASE2025_SELD_dataset/video_eval/eval/sample06813.mp4'}}]}\""

# Build 'node1 package'
[group("Build Individual")]
build-talker:
    podman compose exec -it node1 bash -lc "colcon build --packages-select talker_pkg"

# Build 'node2 package'
[group("Build Individual")]
build-listener:
    podman compose exec -it node2 bash -lc "colcon build --packages-select listener_pkg"

# Build 'node3 package'
[group("Build Individual")]
build-seld:
    podman compose exec -it node3 bash -lc "colcon build --packages-select seld_pkg"
