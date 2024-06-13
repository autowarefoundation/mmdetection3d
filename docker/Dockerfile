ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine, MMCV and MMDetection
RUN pip install openmim && \
    mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0rc5, <3.3.0"

# Install MMDetection3D
RUN conda clean --all \
    && git clone https://github.com/autowarefoundation/mmdetection3d.git -b main /mmdetection3d \
    && cd /mmdetection3d \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection3d
