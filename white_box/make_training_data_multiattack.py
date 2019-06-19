from __future__ import print_function
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--multi', type=int, default=100) #100: all 12 datas; 1000: 12,datas 4 models
parser.add_argument('--cnn_model', type=int, default=4)
parser.add_argument('--cnn_epochs', type=int, default=20)
args = parser.parse_args()

if args.dataset==1:
    base_savedir = 'mnist/'
elif args.dataset==2:
    base_savedir = 'fmnist/'

if args.multi == 100:
    savedir1 = base_savedir + 'AdvSave/FGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.2)
    savedir2 = base_savedir + 'AdvSave/FGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.25)
    savedir3 = base_savedir + 'AdvSave/FGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.3)
    savedir4 = base_savedir + 'AdvSave/FGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.35)
    savedir5 = base_savedir + 'AdvSave/RANDFGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.2)
    savedir6 = base_savedir + 'AdvSave/RANDFGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.25)
    savedir7 = base_savedir + 'AdvSave/RANDFGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.3)
    savedir8 = base_savedir + 'AdvSave/RANDFGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.35)
    savedir9 = base_savedir + 'AdvSave/CW/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.2)
    savedir10 = base_savedir + 'AdvSave/CW/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.25)
    savedir11 = base_savedir + 'AdvSave/CW/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.3)
    savedir12 = base_savedir + 'AdvSave/CW/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.35)

    xt = np.load(savedir1 + '/xt.npy')
    yt = np.load(savedir1 + '/yt.npy')
    adv_x1 = np.load(savedir1 + 'adv_x.npy')
    adv_x2 = np.load(savedir2 + 'adv_x.npy')
    adv_x3 = np.load(savedir3 + 'adv_x.npy')
    adv_x4 = np.load(savedir4 + 'adv_x.npy')
    adv_x5 = np.load(savedir5 + 'adv_x.npy')
    adv_x6 = np.load(savedir6 + 'adv_x.npy')
    adv_x7 = np.load(savedir7 + 'adv_x.npy')
    adv_x8 = np.load(savedir8 + 'adv_x.npy')
    adv_x9 = np.load(savedir9 + 'adv_x.npy')
    adv_x10 = np.load(savedir10 + 'adv_x.npy')
    adv_x11 = np.load(savedir11 + 'adv_x.npy')
    adv_x12 = np.load(savedir12 + 'adv_x.npy')

    adv_x = np.concatenate([adv_x1, adv_x2, adv_x3,adv_x4, adv_x5, adv_x6,adv_x7, adv_x8, adv_x9,adv_x10, adv_x11, adv_x12], axis=0)
    xt = np.concatenate([xt, xt, xt,xt, xt, xt,xt, xt, xt,xt, xt, xt], axis=0)
    yt = np.concatenate([yt, yt, yt,yt, yt, yt,yt, yt, yt,yt, yt, yt], axis=0)

    savedir = base_savedir + 'AdvSave/multiattack/model_{}/cnnEpochs_{}/multi_{}/'.format(args.cnn_model, args.cnn_epochs, args.multi)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    np.save(savedir + 'adv_x.npy', adv_x)
    np.save(savedir + 'xt.npy', xt)
    np.save(savedir + 'yt.npy', yt)

