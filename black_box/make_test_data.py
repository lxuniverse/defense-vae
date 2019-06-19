from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import *
from cnn_models import model_a, model_b, model_c, model_d, model_e
import os
import math
import argparse
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1)###1:mnist; 2:fmnist
parser.add_argument('--cnn_model_sub', type=int, default=1)
parser.add_argument('--cnn_model_bb', type=int, default=1)
parser.add_argument('--attack_method', type=int, default=1)### 1:fgsm; 2:rand fgsm 3:CW
parser.add_argument('--epsilon', type=float, default=0.3)###for cw: 0.2 without feed; 0.3 with feed
parser.add_argument('--cnn-epochs', type=int, default=20)
args = parser.parse_args()

if args.dataset==1:
    data = input_data.read_data_sets('../data/mnist',validation_size=0)
    base_savedir = 'mnist/'
elif args.dataset==2:
    data = input_data.read_data_sets('../data/fashion',validation_size=0)
    base_savedir = 'fmnist/'

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

X_train = data.train.images
X_train = np.reshape(X_train,(60000,28,28,1))
Y_train = data.train.labels
Y_train = to_categorical(Y_train, num_classes=10)
X_test= data.test.images
X_test = np.reshape(X_test,(10000,28,28,1))
Y_test= data.test.labels
Y_test = to_categorical(Y_test, num_classes=10)
cnn_dir = '../cnn_models/' + base_savedir + 'bb_{}_sub_{}/'.format(args.cnn_model_bb, args.cnn_model_sub)

batch_size=50
config_args = {}

#########
x_shape, classes = [28,28,1], 10
nb_classes = classes
type_to_bbmodels = {1: model_a, 2: model_b, 3: model_c, 4: model_d}
bb_model = type_to_bbmodels[args.cnn_model_bb](
        input_shape=[None] + x_shape, nb_classes=10,
    )
type_to_submodels = {1: model_b, 2: model_e}

images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))
rng = np.random.RandomState([11, 24, 1990])
tf.set_random_seed(11241990)

sub_model = type_to_submodels[args.cnn_model_sub](
    input_shape=[None] + x_shape, nb_classes=10,
)
model = bb_model
##########
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, cnn_dir + "model.ckpt")
    print("Model restored.")

    if args.attack_method == 1:
        attack_params = {'eps': args.epsilon, 'clip_min': 0., 'clip_max': 1.}
        attack_obj = FastGradientMethod(sub_model, sess=sess)
        savedir = base_savedir + 'AdvSave_test/FGSM/modelbb_{}/modelsub_{}/cnnEpochs_{}/epsilon_{}/'.format(args.cnn_model_bb, args.cnn_model_sub, args.cnn_epochs, args.epsilon)

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    adv_x = attack_obj.generate(images_tensor, **attack_params)
    preds_adv = sub_model.get_probs(adv_x)
    eval_par = {'batch_size': batch_size}
    nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
    assert nb_batches * batch_size >= len(X_test)
    X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
    adv_l = []
    for batch in range(nb_batches):
        start = batch * batch_size
        end = min(len(X_test), start + batch_size)
        cur_batch_size = end - start
        X_cur[:cur_batch_size] = X_test[start:end]
        Y_cur[:cur_batch_size] = Y_test[start:end]
        feed_dict = {images_tensor: X_cur}
        cur_adv = adv_x.eval(feed_dict=feed_dict)
        adv_l.append(cur_adv)
    adv_examples = np.vstack(adv_l)[0:10000]

np.save(savedir + 'adv_x.npy', adv_examples)
np.save(savedir + 'xt.npy', X_test)
np.save(savedir + 'yt.npy', Y_test)








