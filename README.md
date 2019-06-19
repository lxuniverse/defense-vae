# defense_vae
This repository contains the implementation of our [paper](https://arxiv.org/abs/1812.06570).

## Dependences
Python 2.7, Pytorch, Tensorflow 1.7, Cleverhans 2.1.0

## Repo Structure

* white_box: scripts for generating adversarial images for training and testing data, training VAE, reconstructing testing adversarial images, testing white-box defense accuracy. 
* black_box: scripts for generating black-box adversarial images testing data, reconstructing black-box testing adversarial images, testing black-box defense accuracy.
* data: folder to place raw image data.
* cnn_models: pretrained CNN models as the attacking target and pretrained substitute models for black-box attacks. 

## Quick Start

* Clone the Repo and install python libraries used in the code.

* Download the datasets with:
    * cd data
    * sh data/download_datasets.sh

* For the white-box defense, please run:
    * cd white_box
    * python main_script.py --gpu 0 --dataset 1 --cnn_model 1 
        * --gpu: choose GPU index, use 0 if you only have one.
        * --dataset: 1 or 2. 1 is MNIST and 2 is FMNIST.
        * --cnn_model: 1 ~ 4. corresponding to model a~d. See the paper for details.

* For the black-box defense, please run:
    * cd black_box
    * python main_script.py --gpu 0 --dataset 1 --cnn_model_bb 1 --cnn_model_sub 1 
        * --gpu: choose GPU index, use 0 if you only have one.
        * --dataset: 1 or 2. 1 is MNIST and 2 is FMNIST.
        * --cnn_model_bb: 1 ~ 4. choose the model to be attacked, which corresponding to model a~d. See the paper for details.
        * --cnn_model_sub: 1 or 2. corresponding to model b or e. See the paper for details.

* The final results will be written in the results.txt in white_box and black_box folders. The attack index 1,2,3 is corresponding to FGSM, RANDFGSM and CW attack.

* Please run the black-box experiment after white-box experiment, because black-box experiment need the VAE model trained in the white-box defense process. The whole experiment need about half to one hour, depend on the hardware you use.

## Citations

Please cite our paper if it helps you:
    
    @article{li2018defense,
      title={Defense-VAE: A Fast and Accurate Defense against Adversarial Attacks},
      author={Li, Xiang and Ji, Shihao},
      journal={arXiv preprint arXiv:1812.06570},
      year={2018}
    }

## Contacts

Xiang Li, xli62@student.gsu.edu

