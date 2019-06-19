from __future__ import print_function
import argparse
import torch.utils.data
from torch import optim
import os
from vae_models import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--conv_mu', type=float, default=0.0)
parser.add_argument('--conv_sigma', type=float, default=0.02)
parser.add_argument('--bn_mu', type=float, default=1.0)
parser.add_argument('--bn_sigma', type=float, default=0.02)
parser.add_argument('--r_begin', type=float, default=0.0)
parser.add_argument('--r_end', type=float, default=0.0)
parser.add_argument('--sigmoid', type=bool, default=0)
parser.add_argument('--sigmoid_lambda', type=float, default=10)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--optim', type=int, default=1)
parser.add_argument('--vae_model', type=int, default=18)
parser.add_argument('--multi', type=int, default=100)
parser.add_argument('--cnn_model', type=int, default=4)
parser.add_argument('--cnn_epochs', type=int, default=20)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--attack_method', type=int, default=1)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.dataset==1:
    base_savedir = 'mnist/'
elif args.dataset==2:
    base_savedir = 'fmnist/'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(args.conv_mu, args.conv_sigma)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(args.bn_mu, args.bn_sigma)
        m.bias.data.fill_(0)

if args.sigmoid:
    r_interval = (args.r_end - args.r_begin) / args.epochs
    r_list = np.arange(args.r_begin, args.r_end, r_interval)
    r_list = (r_list - args.r_end / 2) * args.sigmoid_lambda
    r_list = 1 / (1 + np.exp(-r_list))
else:
    if args.r_end == args.r_begin:
        r_list = [args.r_end] * args.epochs
    else:
        r_interval = (args.r_end - args.r_begin) / args.epochs
        r_list = np.arange(args.r_begin, args.r_end, r_interval)

### read adv and true
savedir = base_savedir + 'AdvSave/multiattack/model_{}/cnnEpochs_{}/multi_{}/'.format(args.cnn_model, args.cnn_epochs, args.multi)
adv_x = np.load(savedir + 'adv_x.npy')
adv_x = adv_x.transpose(0, 3, 1, 2)
adv_x = torch.tensor(adv_x)

xt = np.load(savedir + 'xt.npy')
xt = xt.transpose(0, 3, 1, 2)
xt = torch.tensor(xt).type(torch.FloatTensor)
yt = np.load(savedir + 'yt.npy')
yt = torch.tensor(yt)
yt = yt.to(device)

train_data = torch.utils.data.TensorDataset(adv_x, xt)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

def train(epoch, savedir):
    model.train()
    train_loss = 0

    loss_ll = []
    loss_B = []
    loss_K = []
    for batch_idx, (adv_batch, clean_batch) in enumerate(train_data_loader):
        adv_batch = adv_batch.to(device)
        clean_batch = clean_batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(adv_batch)

        r = torch.tensor(r_list[epoch - 1]).to(device)
        loss, BCE, KLD = loss_lambda(recon_batch, clean_batch, mu, logvar, r)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(adv_batch), len(train_data_loader.dataset),
                       100. * batch_idx / len(train_data_loader), loss.item() / len(adv_batch)))
        loss_a = loss.item() / len(adv_batch)
        loss_ll.append(loss_a)

        loss_b = BCE.item() / len(adv_batch)
        loss_k = KLD.item() / len(adv_batch)
        loss_B.append(loss_b)
        loss_K.append(loss_k)

    loss_norm = train_loss / len(train_data_loader.dataset)

    torch.save(model.state_dict(), savedir + 'model{}.pth'.format(epoch))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_norm))
    return loss_ll, loss_B, loss_K

model_dict = {18: VAE18()}
vae_num = args.vae_model
model = model_dict[vae_num].to(device)

model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

savedir = base_savedir + 'vae_model_lambda/init_multi/vaeModel_{}/cnnModel_{}/epochs_{}/lr_{}/batchSize_{}' \
                         '/multi_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/init1_{}/int2_{}/init3_{}/int4_{}/'.format(
    args.vae_model, args.cnn_model, args.epochs, args.lr, args.batch_size, args.multi, args.r_begin, args.r_end,
    args.sigmoid, args.sigmoid_lambda,args.conv_mu,args.conv_sigma,args.bn_mu,args.bn_sigma)

if not os.path.exists(savedir):
    os.makedirs(savedir)

loss_l = []
loss_bbb = []
loss_kkk = []
for epoch in range(1, args.epochs + 1):
    loss, loss1, loss2 = train(epoch, savedir)
    loss_l.append(loss)
    loss_bbb.append(loss1)
    loss_kkk.append(loss2)

loss_numpy = np.hstack(loss_l)
loss_numpy1 = np.hstack(loss_bbb)
loss_numpy2 = np.hstack(loss_kkk)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(range(loss_numpy.shape[0]), loss_numpy)
fig.savefig(savedir + 'loss.png')
plt.close(fig)
np.save(savedir + 'loss', loss_numpy)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(range(loss_numpy2.shape[0]), loss_numpy2)
fig.savefig(savedir + 'loss_k.png')
plt.close(fig)
np.save(savedir + 'loss_k', loss_numpy2)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(range(loss_numpy1.shape[0]), loss_numpy1)
fig.savefig(savedir + 'loss_b.png')
plt.close(fig)
np.save(savedir + 'loss_b', loss_numpy1)

torch.save(model.state_dict(), savedir + 'model.pth')
