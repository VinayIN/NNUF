# Use ROS 2 Jazzy base (Python 3.10, Ubuntu 24.04)
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

RUN which pip && pip -V && python3 -c "import sys; print(sys.prefix)"

# Upgrade pip in venv first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# ROS 2 Setup
RUN echo "source /opt/ros/jazzy/setup.bash" > /etc/profile.d/ros2.sh \
    && chmod +x /etc/profile.d/ros2.sh

ENTRYPOINT ["/bin/bash", "--login"]

CMD []