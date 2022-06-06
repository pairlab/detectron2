FROM nvidia/cudagl:11.3.0-devel-ubuntu18.04

# install basic libs
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y &&\
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    software-properties-common 
    

WORKDIR /pointrend

# set up conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O ./anaconda.sh &&\
  /bin/bash ./anaconda.sh -b -p /opt/conda &&\
  /opt/conda/bin/conda init bash &&\
  rm ./anaconda.sh

RUN /opt/conda/bin/conda create -n pointrend python=3.7
RUN /opt/conda/bin/conda run -n pointrend pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN /opt/conda/bin/conda run -n pointrend pip install pyyaml==5.1 opencv-python matplotlib
RUN git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
RUN /opt/conda/bin/conda run -n pointrend pip install -e detectron2_repo

# update gcc 
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:ubuntu-toolchain-r/test
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y &&\
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  gcc-9  g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9


