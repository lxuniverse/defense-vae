import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=int, default=2)
parser.add_argument('--cnn_model_bb', type=int, default=1)
parser.add_argument('--cnn_model_sub', type=int, default=1)
args = parser.parse_args()

GPU = args.gpu
dataset = args.dataset
attack_method = 1  # For black-box defense, we only support FGSM (1)
cnn_model_bb = args.cnn_model_bb
cnn_model_sub = args.cnn_model_sub

# Generate testing data
print('Start generating adversarial images for testing')

for eps in [0.3]:
    os.system("CUDA_VISIBLE_DEVICES={} python make_test_data.py "
              "--dataset {} --attack_method {} --cnn_model_bb {} --cnn_model_sub {} --epsilon {}"
              .format(GPU, dataset, attack_method, cnn_model_bb, cnn_model_sub, eps))

# Reconstruct the testing data
print('Start reconstructing adversarial testing images')
os.system("CUDA_VISIBLE_DEVICES={} python rec_multi_fromwhite.py "
          "--dataset {} --cnn_model_bb {} --cnn_model_sub {}"
          .format(GPU, dataset, cnn_model_bb, cnn_model_sub))

# defense results
print('Defense Results:')
for eps in [0.3]:
    os.system("CUDA_VISIBLE_DEVICES={} python defense_multi.py "
              "--dataset {} --attack_method {} --cnn_model_bb {} --cnn_model_sub {} --test_epsilon {}"
              .format(GPU, dataset, attack_method, cnn_model_bb, cnn_model_sub, eps))