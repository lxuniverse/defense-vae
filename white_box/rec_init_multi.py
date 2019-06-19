from __future__ import print_function
import argparse
import numpy as np
import os
from vae_models import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--conv_mu', type=float, default=0.0)
parser.add_argument('--conv_sigma', type=float, default=0.02)
parser.add_argument('--bn_mu', type=float, default=1.0)
parser.add_argument('--bn_sigma', type=float, default=0.02)
parser.add_argument('--r_begin', type=float, default=0.0)
parser.add_argument('--r_end', type=float, default=0.0)
parser.add_argument('--sigmoid', type=bool, default=0)
parser.add_argument('--sigmoid_lambda', type=float, default=10)
parser.add_argument('--vae_model', type=int, default=18)
parser.add_argument('--multi', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--model_epochs', type=int, default=5)
parser.add_argument('--optim', type=int, default=1)
parser.add_argument('--test_epsilon', type=float, default=0.3)
parser.add_argument('--attack_method', type=int, default=1)
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--cnn_epochs', type=int, default=20)
args = parser.parse_args()

if args.dataset==1:
    base_savedir = 'mnist/'
elif args.dataset==2:
    base_savedir = 'fmnist/'

device = torch.device("cpu")
model_dict = {18: VAE18()}
vae_num = args.vae_model
model_vae = model_dict[vae_num].to(device)

load_dir = base_savedir + 'vae_model_lambda/init_multi/vaeModel_{}/cnnModel_{}/epochs_{}/lr_{}/batchSize_{}/multi_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/init1_{}/int2_{}/init3_{}/int4_{}/'.format(
    args.vae_model, args.cnn_model, args.train_epochs,
    args.lr, args.batch_size, args.multi, args.r_begin, args.r_end, args.sigmoid, args.sigmoid_lambda,
    args.conv_mu, args.conv_sigma, args.bn_mu, args.bn_sigma)

model_vae.load_state_dict(torch.load(load_dir + 'model{}.pth'.format(args.model_epochs)))
model_vae.to(device)

##load test data
if args.attack_method == 1:
    savedir = base_savedir + 'AdvSave_test/FGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, 20, args.test_epsilon)
elif args.attack_method == 2:
    savedir = base_savedir + 'AdvSave_test/RANDFGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, 20, args.test_epsilon)
elif args.attack_method == 3:
    savedir = base_savedir + 'AdvSave_test/CW/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, args.test_epsilon)
adv_x = np.load(savedir + 'adv_x.npy')
adv_x = adv_x.transpose(0, 3, 1, 2)
adv_x = torch.tensor(adv_x)
adv_x = adv_x.to(device)

xt = np.load(savedir + '/xt.npy')
xt = xt.transpose(0, 3, 1, 2)
xt = torch.tensor(xt).type(torch.FloatTensor).to(device)
yt = np.load(savedir + '/yt.npy')
adv_acc = np.load(savedir + '/adv_acc.npy')

### reconstruct
model_vae.training = False
rec_adv_x, mu, logvar = model_vae(adv_x)

####save
rec_adv_x_numpy = rec_adv_x.cpu().detach().numpy()
if args.attack_method ==1:
    rec_dir = base_savedir + 'rec_lambda/init_multi/vaeModel_{}/cnnModel_{}/train_epoch_{}/model_epoch_{}/lr_{}/batchSize_{}/multi_{}/testEps_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/'.format(
        args.vae_model, args.cnn_model, args.train_epochs, args.model_epochs,
        args.lr, args.batch_size,args.multi, args.test_epsilon,args.r_begin,args.r_end,args.sigmoid,args.sigmoid_lambda)
elif args.attack_method ==2:
    rec_dir = base_savedir + 'rec_lambda/init2_multi/vaeModel_{}/cnnModel_{}/train_epoch_{}/model_epoch_{}/lr_{}/batchSize_{}/multi_{}/testEps_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/'.format(
        args.vae_model, args.cnn_model, args.train_epochs, args.model_epochs,
        args.lr, args.batch_size,args.multi, args.test_epsilon,args.r_begin,args.r_end,args.sigmoid,args.sigmoid_lambda)
elif args.attack_method ==3:
    rec_dir = base_savedir + 'rec_lambda/init3_multi/vaeModel_{}/cnnModel_{}/train_epoch_{}/model_epoch_{}/lr_{}/batchSize_{}/multi_{}/testEps_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/'.format(
        args.vae_model, args.cnn_model, args.train_epochs, args.model_epochs,
        args.lr, args.batch_size,args.multi, args.test_epsilon,args.r_begin,args.r_end,args.sigmoid,args.sigmoid_lambda)

if not os.path.exists(rec_dir):
    os.makedirs(rec_dir)
np.save(rec_dir + 'rec', rec_adv_x_numpy)



