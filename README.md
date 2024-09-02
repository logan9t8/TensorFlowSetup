# TensorFlow Installation in WSL/Linux with GPU support (NVIDIA)

**Note: TensorFlow does not have native support for GPU on Windows**

## Conda Installation

### Arch Linux
`yay -S miniconda3`

### Generic Linux
```shell
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

## TensorFlow Installation

### CPU alone

**Note: The TensorFlow version should be compatible with the python versions\
The mapping of TensorFlow and supported python versions can be found [here](https://www.tensorflow.org/install/source#cpu)**

```shell
conda create --name tf python=3.12
conda activate tf
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip
pip install tensorflow==2.17.0
```

### With GPU support
**Note: The TensorFlow version should be compatible with the CUDA and cuDNN versions\
The mapping of TensorFlow, CUDA, and cuDNN versions can be found [here](https://www.tensorflow.org/install/source#gpu)**

```shell
conda install --channel conda-forge cudatoolkit=12.3
pip install nvidia-cudnn-cu11==8.9
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Suppress TensorRT warnings
echo 'export TF_CPP_MIN_LOG_LEVEL="2"' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Verification

```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# Verify CPU
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Keras (if required)
Keras is no longer a separate package and is included in TensorFlow
```python
from tensorflow import keras
```

## Credits
* https://docs.anaconda.com/miniconda
* https://www.tensorflow.org
* https://developer.nvidia.com/cuda-toolkit
* https://developer.nvidia.com/cudnn
