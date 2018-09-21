# TensorFlow implementation of Generative Adversarial Networks
- This repository contains tensorflow implementations of GANs inspired by several other repositories of GANs or generative models ([generative-models](https://github.com/wiseodd/generative-models), [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)).
- The aim of this repository is for me to learn and experiment on various GANs.
- I will try to keep the architecture of each GAN the same as the original paper.

# Models
*Name* | *Description* | *Paper* | *Details* |
:--: | :---: | :--: | :---: | 
DCGAN | | [paper](https://arxiv.org/abs/1511.06434) | [details](docs/dcgan/)
LSGAN | | [paper](https://arxiv.org/abs/1611.04076) | [details](docs/lsgan/)
BEGAN | | [paper](https://arxiv.org/abs/1703.10717) | [details](docs/began/)

# Usage
The script [`example/gans.py`](example/gans.py)

### Prepare

### Argument
* `--train`: Train the model.
* `--generate`: Randomly sample images from trained model.
* `--load`: The epoch ID of pre-trained model to be restored.
* `--gan_type`: Type of GAN for experiment. Default: `dcgan`. Other options: `lsgan`, `began`.
* `--dataset`: Dataset used for experiment. Default: `mnist`. Other options: `celeba`.
* `--zlen`: Length of input random vector z. Default: `100`
* `--lr`: Initial learning rate. Default: `2e-4`.
* `--keep_prob`: Keep probability of dropout. Default: `1.0`.
* `--bsize`: Batch size. Default: `128`.
* `--maxepoch`: Max number of epochs. Default: `50`.
* `--ng`: Number of times of training generator for each step. Default: `1`.
* `--nd`: Number of times of training distrminator for each step. Default: `1`.

### Train the model

### Sample images from trained model

# Result
*Name* | *MNIST* |*CelebA* |
:--: | :---: | :--: |
DCGAN |
LSGAN |
BEGAN |
