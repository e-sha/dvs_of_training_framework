from nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
LABEL maintainer="shalnov.eugen@gmail.com"

ENV ROOT /app
ENV DEBIAN_FRONTEND noninteractive

# Set the working directory to ROOT
WORKDIR ${ROOT}

# install pip
RUN apt update && apt install -y python3-pip \
  libopencv-dev \
  python3-setuptools \
  python3-opencv \
  libhdf5-dev \
  vim \
  cmake \
  libeigen3-dev \
  git

# copy files
ADD requirements.txt ./
# install python packages
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

# build modules
RUN bash build.sh && rm -rf utils/modules_to_build

## copy files
#ADD sliding_window.py dvs_visualizer.py video_logger.py tracker.py ./
#ADD stream-receiver ./stream-receiver
#ADD identification ./identification
#ADD rpg_e2vid ./rpg_e2vid
#ADD ssd_detector ./ssd_detector
#ADD redis_tools.py gui.py test.py config.yml dvs_reader.py identifier.py resizer.py graph_config.yml ./
#
## start tracking
#CMD ["python3", "gui.py"]
