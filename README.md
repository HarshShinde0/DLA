# Deep Learning AIO - DLA (GPU Enabled)
[![Deep Learning AIO - DLA](https://github.com/HarshShinde0/DLA/actions/workflows/docker-run.yml/badge.svg)](https://github.com/HarshShinde0/DLA/actions/workflows/docker-run.yml)

This Docker image (`harshshinde/deep-learning-aio:gpu`) provides a fully equipped environment for deep learning and AI model development with GPU support. It comes preconfigured with a wide range of popular machine learning frameworks, including TensorFlow, PyTorch, Keras, MXNet, and many others, ensuring you can seamlessly build, train, and deploy AI models on a GPU-enabled system.

Leveraging NVIDIA CUDA 11.8 and cuDNN 8, this image is optimized for performance and acceleration on systems with NVIDIA GPUs. Whether you're working on computer vision, natural language processing, or other deep learning tasks, this environment provides all the tools and libraries you need.

## Key Features

- **GPU Acceleration with CUDA 11.8 & cuDNN 8**: Optimized for running deep learning models on NVIDIA GPUs, this image supports all major deep learning frameworks with CUDA-backed performance enhancements.

- **Comprehensive AI & ML Libraries**:
  - **TensorFlow**: For deep learning model development and training.
  - **PyTorch & TorchVision/Torchaudio**: For dynamic neural networks and media processing.
  - **Keras, MXNet, Caffe, Chainer, Sonnet**: Popular frameworks for building and optimizing neural networks.
  - **Scikit-learn, Matplotlib, Pandas, Numpy**: Essential libraries for data preprocessing, visualization, and model evaluation.

- **Preconfigured Jupyter Notebook & JupyterLab**: The image comes ready to run Jupyter Notebooks and JupyterLab, enabling interactive development. The notebook server is configured to run on all IPs, allowing access from any machine on the network.

- **Pre-installed Dependencies**: Includes essential libraries such as `build-essential`, `git`, `curl`, `python3-pip`, `python3-dev`, and more. This ensures that you can easily install additional libraries or dependencies as needed.

- **Miniconda Package Manager**: Miniconda is pre-installed for managing Python packages and environments easily.

## Included Libraries and Frameworks

- **Deep Learning Frameworks**: TensorFlow, PyTorch, Keras, MXNet, Chainer, Lasagne, Caffe, Sonnet
- **Data Science & ML Libraries**: Scikit-learn, Pandas, Matplotlib, Numpy
- **Jupyter Tools**: Jupyter, JupyterLab, Notebook

## Usage

### Running the Docker Container

To start a container from this image with GPU support, run the following command:

```bash
docker run -d -p 8888:8888 --gpus all --restart unless-stopped harshshinde/deep-learning-aio:gpu
