# Defense-VAE
This repository contains the implementation of [defense-vae](https://arxiv.org/abs/1812.06570).

## Dependences
Python 2.7, Pytorch, Tensorflow 1.7, Cleverhans 2.1.0

## Repo Structure

* white_box: scripts to generate adversarial images from training and test data, train VAE, reconstruct from adversarial images, and test white-box defense accuracy. 
* black_box: scripts to generate adversarial images from test data, reconstruct from adversarial images, and test black-box defense accuracy.
* data: folder to place raw image data.
* cnn_models: pretrained CNN models as the attacking targets and pretrained substitute models for black-box attacks. 

## Quick Start

* Clone the Repo

* Download the datasets:
    * cd data
    * sh download_datasets.sh

* For the white-box defense, please run:
    * cd white_box
    * python main_script.py --gpu 0 --dataset 1 --cnn_model 1 
        * --gpu: choose GPU index, use 0 if you only have one.
        * --dataset: 1 or 2, where 1 is for MNIST and 2 is for FMNIST.
        * --cnn_model: 1 ~ 4 corresponding to model a~d. See the paper for details.

* For the black-box defense, please run:
    * cd black_box
    * python main_script.py --gpu 0 --dataset 1 --cnn_model_bb 1 --cnn_model_sub 1 
        * --gpu: choose GPU index, use 0 if you only have one.
        * --dataset: 1 or 2, where 1 is for MNIST and 2 is for FMNIST.
        * --cnn_model_bb: 1 ~ 4, choose the model to be attacked, which corresponds to model a~d. See the paper for details.
        * --cnn_model_sub: 1 or 2 corresponding to model b or e. See the paper for details.

* The final results will be written into results.txt in the white_box and black_box folders. The attack index 1,2,3 corresponds to the FGSM, RANDFGSM and CW attack.

* Please run the black-box experiments after the white-box experiments because the black-box experiments need the VAE models trained in the white-box experiments. 

## Citation

If you found this paper useful, please cite it.
    
    @inproceedings{defense-vae19,
      title     = {Defense-VAE: A Fast and Accurate Defense Against Adversarial Attacks},
      author    = {Xiang Li and Shihao Ji},
      booktitle = {Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
      publisher = {Springer},
      pages     = {191--207},
      year      = {2019}
    }

## Contact

Xiang Li, xli62@student.gsu.edu

