# Context-Cluster

Context-Cluster is entirely new paradigm that considers image as Set of Points and generates representations. This repository contains ready-to train models & visualize clusters over images. This repository utilizes original [Context-Cluster](https://github.com/ma-xu/Context-Cluster) ([openreview](https://openreview.net/pdf?id=awnvqZja69)) implementation. Moreover, we have created helper scripts to train the models and Gradio app to generate cluster visualization for any image.


## Run in a Free GPU powered Gradient Notebook
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/ashutosh1919/context-cluster-pytorch?machine=Free-GPU)


## Setup

The file `installations.sh` contains all the necessary code to install required things. Note that your system must have CUDA to train Context-Cluster models. Also, you may require different version of `torch` based on the version of CUDA. If you are running this on [Paperspace](https://www.paperspace.com/), then the default version of CUDA is 11.6 which is compatible with this code. If you are running it somewhere else, please check your CUDA version using `nvcc --version`. If the version differs from ours, you may want to change versions of PyTorch libraries in the first line of `installations.sh` by looking at [compatibility table](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions).

To install all the dependencies, run below command:

```bash
bash installations.sh
```

Above command also clones the original [Context-Cluster](https://github.com/ma-xu/Context-Cluster) repository into `context_cluster` directory so that we can utilize the original model implementation for training & inference.


## Downloading datasets & Start training (Optional)

`datasets` directory in this repo contains necessary scripts to download the data and make it ready for training. Currently, this repository supports downloading [ImageNet](https://www.image-net.org/) dataset that original authors used.

We have already setup bash scripts for you which will automatically download the dataset for you and will start the training. `train.sh` contains the code which will download the training & validation data to `dataset` directory and will start training the model.

These bash scripts are compatible for Paperspace workspace. But if you are running it elsewhere, then you will need to replace base path of the paths mentioned in these task files.

Before you start the training, you can check & customize all the model arguments in `args.yaml` file. Especially, you may want to change the argument `model` to one of the following: `coc_tiny`, `coc_tiny_plain`, `coc_small`, `coc_medium`. These models differ by the number of layers (stages).

To download data files and start training, you can execute below command:

```bash
bash train.sh
```

Note that the generated checkpoints for the trained model will be available in `context_cluster/outputs` directory. You will need to move `checkpoint.pth.tar` file to `checkpoints` directory for inference at the end of training.

Don't worry if you don't want to train the model. Below section illustrates downloading the pretrained checkpoints for inference.


## Running Gradio Demo

Python script `app.py` contains Gradio demo which lets you visualize clusters on the image. But before we do that, we need to download the pretrained checkpoints into `checkpoints` directory.

To download existing checkpoints, run below command:

```bash
bash checkpoints/fetch_pretrained_checkpoints.sh
```

Note that the latest version of code only has the pretrained checkpoints for `coc_tiny_plain` model type. But you can add the code in `fetch_pretrained_checkpoints.sh` whenever the new checkpoints for other model types are available in [original repository](https://github.com/ma-xu/Context-Cluster).

Now, we are ready to launch Gradio demo. Run followling command to launch demo:

```bash
gradio app.py
```

Open the link in the browser and now you can generate inference from any of available models in `checkpoints` directory. Moreover, you can generate cluster visualization of specific `stage`, `block` and `head`. Upload your image and hit the Submit button.

You should be able to generate cluster visualization for any image as shown below:

<img width="1238" alt="demo" src="https://user-images.githubusercontent.com/20843596/228104200-aed8b4dc-ebce-4d41-b097-34d63fe7f993.png">

Hurray! 🎉🎉🎉  We have created demo to visualize clusters over any image by inferring Context-Cluster model.


## Original Code

`context_cluster` directory contains the original code taken from [Context-Cluster](https://github.com/ma-xu/Context-Cluster) repository. The code present in this directory is exactly same as the original code.


## Reference

Image as Set of Points -- https://openreview.net/forum?id=awnvqZja69

```
@inproceedings{
ma2023image,
title={Image as Set of Points},
author={Xu Ma and Yuqian Zhou and Huan Wang and Can Qin and Bin Sun and Chang Liu and Yun Fu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=awnvqZja69}
}
```

## License

See the [LICENSE](LICENSE) file.
