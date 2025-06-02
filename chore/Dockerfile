FROM ubuntu:22.04

# Set Python version as build argument
ARG PYTHON_VERSION=3.11

# Environment variables for build and runtime
ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=TRUE

# Install system dependencies
RUN apt update &&\
    apt install -y htop tmux vim libsm6 libxext6 libxrender-dev \
                   libgl1-mesa-glx git wget libssl-dev libopencv-dev \
                   libspdlog-dev gcc g++ ffmpeg cmake iputils-ping curl

WORKDIR /hzztai

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=${PATH}:/opt/conda/bin

# Configure conda to use Tsinghua mirror for faster downloads in China
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ &&\
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ &&\
    conda config --set show_channel_urls yes

# Install Python and basic scientific packages
RUN conda clean -ya &&\
    conda install -y python=${PYTHON_VERSION} conda-build &&\
    conda install -y -n base ipykernel --update-deps --force-reinstall &&\
    conda install -y pyyaml numpy pillow ipython cython typing typing_extensions mkl mkl-include ninja &&\
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install specialized packages for image processing
RUN conda install -y conda-forge::pydensecrf conda-forge::ncurses conda-forge::openslide-python

# Install utility and ML packages
RUN pip install dill pika paramiko parse openpyxl loguru pandas &&\
    pip install opencv-python lightgbm future nvitop clearml scikit-learn scikit-image tidecv ipywidgets &&\
    pip install albumentations --no-binary qudida,albumentations mmcv mmengine

# Install PyTorch ecosystem
RUN pip install --upgrade torch torchvision torchaudio xformers

# Install model optimization and deployment tools
RUN pip install tensorrt onnx onnxscript onnxruntime-gpu onnxsim &&\
    pip install omegaconf fvcore submitit tensorboard kornia

# Install Hugging Face and DeepSpeed for transformer models and distributed training
RUN pip install huggingface_hub \
                transformers datasets diffusers[torch] tokenizers evaluate timm \
                accelerate optimum \
                deepspeed

# Install visualization and metrics tools
RUN pip install torchcam umap-learn[plot] torchmetrics swanlab

# Prevent albumentations from auto-updating
ENV NO_ALBUMENTATIONS_UPDATE=true

# Add SDK and set up environment paths
ARG SDK="/hzztai/sdk"
COPY --from=sdk . ${SDK}
ENV PYTHONPATH="${SDK}/reader":"${SDK}":"${SDK}/../":"${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}":\
"${SDK}/reader/hwp":\
"${SDK}/reader/kfbio":\
"${SDK}/reader/kfbf/export_lib":\
"${SDK}/reader/sdpc/so":\
"${SDK}/reader/sdpc/so/ffmpeg":\
"${SDK}/reader/sdpc/so/jpeg"

# Add conda libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib

# Set image metadata
ARG IMAGE_NAME="hzztai/develop"

LABEL Image="${IMAGE_NAME}" \
      Author="laMia" \
      Email="qiang.wang@kfbio.cn" \
      Description="Develop docker image for AI model."

RUN echo "Build HZZTAI develop docker env successfully!"
RUN echo "RUN 'docker save ${IMAGE_NAME}:{YOUR_TAG} | gzip > hzztai-develop.tar.gz' to export this image."
