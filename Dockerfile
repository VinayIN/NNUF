FROM ros:jazzy-ros-base

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_IGNORE_INSTALLED=1

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install -y python3-pip

RUN rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN rm /tmp/requirements.txt

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bash_profile

ENTRYPOINT ["/bin/bash"]
