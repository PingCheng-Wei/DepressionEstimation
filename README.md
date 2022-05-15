# Deep Learning-based Multi-modal Depression Estimation using Knowledge from Micro-facial Expressions, Audio and Text

This is the whole project of my bacherlor thesis.

This repository includes:

* The Source Code of final model, `"Sub-attentional ConvBiLSTM with AVT modality"`
* The Source Code of other models
* Trained weights of each models
* Other useful scripts
* Documentations and images

# DAIC-WOZ Depression Database


# Sub-attentional ConvBiLSTM with AVT modality


# Installation

1. Install Nvidia Driver, CUDA and cuDNN to enable Nvidia GPU
    * Make sure what kind of TensorFlow version you need and install the corresponding version of CUDA and cuDNN. Check out [Tensorflow Build Configurations](https://www.tensorflow.org/install/source#tested_build_configurations). Also take a look at your [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) if you want.
    * Go to [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and download the correct CUDA version you needed. Then run the downloaded file to install. `Remember to add the path variable of the bin folder to enviroment variable in your system path if you try to install on windows`
    * Go to [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) and download the correct cuDNN version you needed. Then run the downloaded file to install. `Remember to add the path variable of the bin folder to enviroment variable in your system path if you try to install on windows`
    * Go to [Nvidia Driver Website](https://www.nvidia.com/Download/index.aspx?lang=en-us) and download the Driver corresponding to your GPU. Then run the downloaded file to install. To find out what GPU you have, run this code:
        ```bash
        lspci | grep VGA
        ```
        you will see something like this ` NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)`
    
    * For more detailed installation process, you could visit the following links:
        * [Install CUDA and CUDNN on Windows & Linux](https://techzizou.com/install-cuda-and-cudnn-on-windows-and-linux/)
        * [CUDA, CuDNN, and Tensorflow installation on windows and Linux](https://codeperfectplus.herokuapp.com/cuda-and-tensorflow-installation-on-windows-linux)
        * [How to Install CUDA on Ubuntu 18.04](https://linoxide.com/install-cuda-ubuntu/)
        * Nvidia Official Guide:
            | |CUDA|cuDNN|
            |---|---|---|
            |Linux|[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)|[Installing cuDNN On Linux](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux)|
            |Windows|[CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)|[Installing cuDNN On Windows](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)|
2. Create a root directory (e.g. SD-MaskRCNN) and clone this repository into it

   ```bash
   git clone </path/to/repository>
   ```

   or you could just download this repository and unzip the downloaded file

3. Install Anaconda from [official link](https://docs.anaconda.com/anaconda/install/index.html)

4. Install dependencies in an enviroment in Anaconda

    import directly the enviroment from the `environment.yml` file. run:

    ```bash
    conda env create --name <where/to/store/this/env+env_name> --file=<path/to/environment.yml/file>
    # for example: conda env create --name ./path/envname --file=environment.yml
    ```

    This will automatically create an enviroment named "SD-MaskRCNN", which includes all the libraries that DepressionEstimation needs

   * if the method above doesn't work, you then have to create a new enviroment and install requirements. After that their might still have some libraries missing. You could only fix this problem by gradually testing and debugging to see what error message you got. It will show what library or module is missing and you then install the library with conda command in this enviroment

     ```bash
      # Please replace "myenv" with the environment name.
      conda create --name myenv
      # go into root directory of DepressionEstimation
      cd <path/to/root/directory>
      # Install requirements
      pip3 install -r requirements.txt
     ```

   * For more infomation about Anaconda enviroments managing, please check this [links](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

5. Download the model weights and store them in the each model in `models/<which model>/model_weights` directory. The followings are the available pretrained weights:

TODO


7. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

   or run this to install this package with conda
   ```
   conda install -c conda-forge pycocotools
   ```


# Implementation

## Test a Model

## Train a New Model


# Results