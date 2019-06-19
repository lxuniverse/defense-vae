from cleverhans.utils_tf import model_eval
from cleverhans_tutorials.tutorial_models import *
import argparse
import numpy as np
from cnn_models import model_a, model_b, model_c, model_d

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=int, default=2)###1:mnist; 2:fmnist

parser.add_argument('--r_begin', type=float, default=0.0)
parser.add_argument('--r_end', type=float, default=0.0)
parser.add_argument('--sigmoid', type=bool, default=0)
parser.add_argument('--sigmoid_lambda', type=float, default=10)

parser.add_argument('--vae_model', type=int, default=18)
parser.add_argument('--train_epsilon', type=float, default=0.3)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--model_epochs', type=int, default=5)

parser.add_argument('--optim', type=int, default=1) #

parser.add_argument('--multi', type=int, default=100)
parser.add_argument('--attack_method', type=int, default=1)
parser.add_argument('--cnn_model', type=int, default=1)
parser.add_argument('--test_epsilon', type=float, default=0.3)
parser.add_argument('--cnn-epochs', type=int, default=20)
parser.add_argument('--vae-epochs', type=int, default=20)
parser.add_argument('--defense_method', type=int, default=1)#1,reconstruct 2,optim
args = parser.parse_args()

if args.dataset==1:
    base_savedir = 'mnist/'
elif args.dataset==2:
    base_savedir = 'fmnist/'

### read recon im
if args.attack_method ==1:
    rec_dir = base_savedir + 'rec_lambda/init_multi/vaeModel_{}/cnnModel_{}/train_epoch_{}/model_epoch_{}/lr_{}/' \
                             'batchSize_{}/multi_{}/testEps_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/'.format(
        args.vae_model, args.cnn_model, args.train_epochs, args.model_epochs,
        args.lr, args.batch_size, args.multi, args.test_epsilon, args.r_begin, args.r_end, args.sigmoid, args.sigmoid_lambda)

if args.attack_method ==2:
    rec_dir = base_savedir + 'rec_lambda/init2_multi/vaeModel_{}/cnnModel_{}/train_epoch_{}/model_epoch_{}/lr_{}/' \
                             'batchSize_{}/multi_{}/testEps_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/'.format(
        args.vae_model, args.cnn_model, args.train_epochs, args.model_epochs,
        args.lr, args.batch_size, args.multi, args.test_epsilon, args.r_begin, args.r_end, args.sigmoid, args.sigmoid_lambda)

if args.attack_method ==3:
    rec_dir = base_savedir + 'rec_lambda/init3_multi/vaeModel_{}/cnnModel_{}/train_epoch_{}/model_epoch_{}/lr_{}/' \
                             'batchSize_{}/multi_{}/testEps_{}/rBegin_{}/rEnd_{}/sigmoid_{}/siglambda_{}/'.format(
        args.vae_model, args.cnn_model, args.train_epochs, args.model_epochs,
        args.lr, args.batch_size, args.multi, args.test_epsilon, args.r_begin, args.r_end, args.sigmoid, args.sigmoid_lambda)

adv_x_recon = np.load(rec_dir + 'rec.npy')
adv_x_recon = adv_x_recon.transpose(0, 2, 3, 1)

### read lable
savedir = base_savedir + 'AdvSave_test/FGSM/model_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model, args.cnn_epochs, 0.3)
y_true = np.load(savedir + 'yt.npy')

nb_epochs=args.cnn_epochs
batch_size=128
learning_rate=0.001
train_params = {'nb_epochs': nb_epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}

config_args = {}
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

cnn_dir = '../cnn_models/' + base_savedir + 'CNN_model_{}/'.format(args.cnn_model)
sess = tf.Session(config=tf.ConfigProto(**config_args))
if args.cnn_model == 1:
    model = model_a()
elif args.cnn_model == 2:
    model = model_b()
elif args.cnn_model == 3:
    model = model_c()
elif args.cnn_model == 4:
    model = model_d()
preds = model.get_probs(x)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, cnn_dir + "model.ckpt")
    print("Model restored.")

    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, adv_x_recon, y_true, args=eval_params)

message = 'defense accuracy on CNN model_{} and attack_{} is: {} \n'.format(args.cnn_model, args.attack_method, acc)
print(message)

with open("results.txt", "a") as myfile:
    myfile.write(message)


