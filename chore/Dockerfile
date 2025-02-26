FROM ubuntu:22.04

ARG PYTHON_VERSION=3.11

ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=TRUE

RUN apt update &&\
    apt install -y htop tmux vim libsm6 libxext6 libxrender-dev \
                   libgl1-mesa-glx git wget libssl-dev libopencv-dev \
                   libspdlog-dev gcc g++ ffmpeg cmake iputils-ping curl

WORKDIR /hzztai

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=${PATH}:/opt/conda/bin

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ &&\
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ &&\
    conda config --set show_channel_urls yes

RUN conda clean -ya &&\
    conda install -y python=${PYTHON_VERSION} conda-build &&\
    conda install -y -n base ipykernel --update-deps --force-reinstall &&\
    conda install -y pyyaml numpy pillow ipython cython typing typing_extensions mkl mkl-include ninja &&\
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda install -y conda-forge::pydensecrf conda-forge::ncurses conda-forge::openslide-python

RUN pip install dill pika paramiko parse openpyxl loguru pandas &&\
    pip install opencv-python lightgbm future nvitop clearml scikit-learn scikit-image tidecv ipywidgets &&\
    pip install albumentations --no-binary qudida,albumentations mmcv mmengine

RUN pip install torch torchvision torchaudio xformers

RUN pip install tensorrt onnx onnxscript onnxruntime-gpu onnxsim &&\
    pip install omegaconf fvcore submitit tensorboard

# intergrate hugging face
RUN pip install accelerate datasets timm transformers

# intergrate embedding visualization
RUN pip install torchcam umap-learn[plot] torchmetrics

ENV NO_ALBUMENTATIONS_UPDATE=true

# ADD SDK & LABEL
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

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib

ARG IMAGE_NAME="hzztai/develop"

LABEL Image="${IMAGE_NAME}" \
      Author="laMia" \
      Email="qiang.wang@kfbio.cn" \
      Description="Develop docker image for AI model."

RUN echo "Build HZZTAI develop docker env successfully!"
RUN echo "RUN 'docker save ${IMAGE_NAME}:{YOUR_TAG} | gzip > hzztai-develop.tar.gz' to export this image."
