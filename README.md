# Multi-modal Depression Estimation based on Sub-attentional Fusion

This is the whole project of my bacherlor thesis. It is about automatic depression estimation based on AI approach. The publication of the paper can be found at [here](https://link.springer.com/chapter/10.1007/978-3-031-25075-0_42) or [here](https://cvhci.anthropomatik.kit.edu/publications_2307.php)

This repository includes:

* The Source Code of final model, `"Sub-attentional ConvBiLSTM with AVT modality"`
* The Source Code of other models
* Trained weights of each model
* Other useful scripts
* Documentations and images

# DAIC-WOZ Depression Database
DAIC-WOZ depression database is utilized in this thesis. Here is the official [link](https://dcapswoz.ict.usc.edu/), where you can send a request and receive your own username and password to access the database.


Some visualizations for each data type exploited in this work.

1. Visual data of micro-facial expression

![Visual data of micro-facial expression](https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/gif_P321_time-58-88.gif)

2. Acoustic data of log-mel spectrogram

![Acoustic data of log-mel spectrogram](https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/mel_spectrogram_comparison.png)

3. Text data of sentence embeddings

![Text data of sentence embeddings](https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/sentence_embeddings.png)


To download the database automatically, there is a created script, `download_DAIC-WOZ.py`, in the `daic_woz_preprocessing` directory for you to use. Please enter this directory and run the script with the following two codes, respectively.

```bash
cd <path/to/daic_woz_preprocessing>

python download_DAIC-WOZ.py --out_dir=<where/to/store/absolute_path> --username=<the_give_username> --password=<the_given_password>
```

After downloading the dataset, if you wish to preprocess the data as we did, two kind of scripts for database generation are provided in the `database_generation_v1` and `database_generation_v2` directory under the folder, `daic_woz_preprocessing`.

* For `database_generation_v1`: This will generate training, validation, test dataset based on train-set, develop-set, test-set, respectively.

    To use this script, please first go into the folder `daic_woz_preprocessing/Excel for splitting data` and find the GT file you need for splitting the data. In our case, it will be "train_split_Depression_AVEC2017.csv", "dev_split_Depression_AVEC2017.csv", and "full_test_split.csv". Or you can also utilize the one provided from the official DAIC_WOZ. Then open the script you want to use and go to the line `if __name__ == '__main__':`, where you can find `# output root`, `# read gt file`, `# get all files path of participant` nearby. Please enter the absolute path to where you want to store, where the GT file is, and where each data type is to each variable. Now you can run the following example code to generate your dataset:

    ```bash
    python <path/to/script/name>
    # for example: python daic_woz_preprocessing/database_generation_v1/database_generation_train.py
    ```

    **Be aware !!! The generated database could be over 200GB since this generated dataset include all of the variable in this thesis for comparison studies in the experiments. Therefore, please exclude the "np.save" parts in the "sliding_window" function, which you don't need. For instance, the one with coordinate+confidence, spectrogram, hog_features, action units, etc. Moreover, you might also don't need the original dataset, which has been tested and demonstrated in this thesis that it is not that useful. Please also exlude all of the code related to it if you don't need it.**

* For `database_generation_v2`: Similar to `database_generation_v1`, but this time train dataset is generated based on "train-set + develop-set" and text dataset is test-set itself.

    To use this script, please first go into the folder `daic_woz_preprocessing/Excel for splitting data` and find the GT file you need for splitting the data. In our case, it will be "full_train_split_Depression_AVEC2017.csv" and "full_test_split.csv". Then for the rest of the steps, please refer to `database_generation_v1` above.

# Sub-attentional ConvBiLSTM with AVT modality

The overall archetecture of the **Sub-attentional ConvBiLSTM** model:
![Sub-attentional_ConvBiLSTM](https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/ConvBiLSTM_Sub-Atten.png)

The CNN layers and BiLSTM blocks are utilized as feature extractors, followed by 8 different attentional late fusion layers with 8 classification heads. For more detail, please refer to the [paper](https://cvhci.anthropomatik.kit.edu/publications_2307.php).

The archetecture of each attentional fusion layer, which includes a gobal attention and a local attention, could be illustrated as below:
![attentional fusion layer](https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/Attentional_Fusion_Block.png)


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

5. Download the model weights and store them in the each model in `models/<which model>/model_weights` directory. The followings are the available pre-trained weights:

    * [AVT and AV pre-trained weights](https://drive.google.com/drive/folders/1f8Ud6hOxjnWJVTpqkhKIV5Ro-XjCyBxV?usp=sharing)

6. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

   or run this to install this package with conda
   ```
   conda install -c conda-forge pycocotools
   ```


# Implementation

To implement the model, please choose a model you desire in the `models` directory first.

```bash
cd models/<desired model>
# for example: cd models/AVT_ConvLSTM_Sub-Attention
```

For each model folder, the following structure can be found

```
<AVT_ConvLSTM_Sub-Attention>/
config/
    config_inference.yaml
    config_phq-subscores.yaml
dataset/
    dataset.py
    utils.py
models/
    convlstm.py
    evaluator.py
    fusion.py
    ...
model_weights/
    <where to store the pretrained weights>
    ...
main_inference.py
main_phq-subscores.py
utils.py
```

If some folders did not exist, e.g. "model_weights", please create it by yourself. As one can notice, each configuration file (config) corresponds to each main script (main_xxxxx.py)and the utility script (utils.py) under the main scripts contains all the local functions for the main scripts. Please make sure the strucutre stay consistence like this, otherwise the model won't work.

## Test a Model

To test a model, please first make sure the data is generated and change all the configuration in the `config_inference.yaml` according to your desire. Also download the pre-trained weights if needed. Then run:

```bash
python main_inference.py
```

For more complex setting, run the following code and set each value to your desire

```bash
python main_inference.py --config_file=<path/to/config.yaml> --device=<'cuda' or 'cpu'> --gpu=<'gpu ID' can be multiple like '2, 3'> --save=<True or False>
# for example: python main_inference.py --config_file=config/config_inference.yaml --device=cuda --gpu=2,3 --save=False
```

## Train a New Model

To train a model, please also first make sure the data is generated and change all the configuration in the `config_phq-subscores.yaml` according to your desire. Then run:

```bash
python main_phq-subscores.py
```

For more complex setting, run the following code and set each value to your desire

```bash
python main_phq-subscores.py --config_file=<path/to/config.yaml> --device=<'cuda' or 'cpu'> --gpu=<'gpu ID' can be multiple like '2, 3'> --save=<True or False>
# for example: python main_phq-subscores.py --config_file=config/config_phq-subscores.yaml --device=cuda --gpu=2,3 --save=True
```

# Results

![visualization_of_recombination](https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/visualization_of_recombination.PNG)
