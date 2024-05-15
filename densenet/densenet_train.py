import argparse, os, sys, math, time
import numpy as np
from DenseNet import DenseNet
from DenseNet2 import DenseNet2
from utils import readData, getDateTime, print_and_write
import evaluation.eval_segm as eval
from scipy.misc.pilutil import imread, imsave
import tensorflow as tf
import itertools
import random
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default='')

parser.add_argument("--train_images", type=str)
parser.add_argument("--train_images_ext", type=str, default='png')
parser.add_argument("--train_labels", type=str, default='')
parser.add_argument("--train_labels_ext", type=str, default='png')

parser.add_argument("--test_images", type=str, default='')
parser.add_argument("--test_images_ext", type=str, default='png')
parser.add_argument("--test_labels", type=str, default='')
parser.add_argument("--test_labels_ext", type=str, default='png')

parser.add_argument("--height", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--gpu_id", type=int, default=0)

parser.add_argument("--save_stitched", type=int, default=1)
parser.add_argument("--eval_every", type=int, default=100)

parser.add_argument('--validate', action='store_false')
parser.add_argument("--val_images", type=str, default="")
parser.add_argument("--val_annotations", type=str, default="")

parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--val_batch_size", type=int, default=2)

parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=-1)

parser.add_argument("--test_start_id", type=int, default=0)
parser.add_argument("--test_end_id", type=int, default=-1)

parser.add_argument("--load_weights", type=str, default="")
parser.add_argument("--load_log", type=str, default="")

parser.add_argument("--index_percent", type=float, default=100)
parser.add_argument("--max_indices", type=int, default=10000)
parser.add_argument("--min_indices", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--loss_type", type=int, default=0)
parser.add_argument("--save_test", type=int, default=0)
parser.add_argument("--gpu_memory_fraction", type=float, default=1.0)
parser.add_argument("--allow_memory_growth", type=int, default=1)
parser.add_argument("--restore_on_nan", type=int, default=1)

parser.add_argument("--preload_images", type=int, default=1)

parser.add_argument("--psi_act_type", type=int, default=0)

parser.add_argument("--n_layers", type=int, default=0)

parser.add_argument("--lr_dec_epochs", type=int, default=10)
parser.add_argument("--lr_dec_rate", type=float, default=0.9)

args = parser.parse_args()

train_images_path = args.train_images
train_images_ext = args.train_images_ext
train_labels_path = args.train_labels
train_labels_ext = args.train_labels_ext
train_batch_size = args.batch_size

test_images_path = args.test_images
test_images_ext = args.test_images_ext
test_labels_path = args.test_labels
test_labels_ext = args.test_labels_ext

n_classes = args.n_classes
height = args.height
width = args.width
validate = args.validate

load_weights = args.load_weights
load_log = args.load_log
end_id = args.end_id
start_id = args.start_id
learning_rate = args.learning_rate
n_epochs = args.n_epochs
index_ratio = args.index_percent / 100.0

gpu_id = args.gpu_id
max_indices = args.max_indices
min_indices = args.min_indices

save_stitched = args.save_stitched
eval_every = args.eval_every

test_start_id = args.test_start_id
test_end_id = args.test_end_id

loss_type = args.loss_type

log_dir = args.log_dir
save_test = args.save_test

restore_on_nan = args.restore_on_nan

preload_images = args.preload_images

psi_act_type = args.psi_act_type

n_layers = args.n_layers

lr_dec_epochs = args.lr_dec_epochs
lr_dec_rate = args.lr_dec_rate

tf.reset_default_graph()

# if gpu_id < 2:
#     if gpu_id < 0:
#         print('Running on CPU')
#         tf_device = '/cpu:0'
#     else:
#         tf_device = '/gpu:{}'.format(gpu_id)
#     with tf.device(tf_device):
#         if n_layers == 0:
#             model = DenseNet(height, width, ch=3, nclass=3, loss_type=loss_type, psi_act_type=psi_act_type)
#         else:
#             model = DenseNet2(n_layers, height, width, ch=3, nclass=3, loss_type=loss_type, psi_act_type=psi_act_type)
# else:

if n_layers == 0:
    model = DenseNet(height, width, ch=3, nclass=n_classes, loss_type=loss_type, psi_act_type=psi_act_type)
else:
    model = DenseNet2(n_layers, height, width, ch=3, nclass=n_classes, loss_type=loss_type, psi_act_type=psi_act_type)

# print('Trainable variables:\n')
trainable_variables = tf.trainable_variables()
n_parameters = 0
n_variables = len(trainable_variables)
for variable in trainable_variables:
    variable_shape = variable.get_shape()
    # print('variable: {}\t shape: {}'.format(variable.name, variable_shape))
    variable_parameters = 1
    for dim in variable_shape:
        variable_parameters *= dim.value

    n_parameters += variable_parameters

print()
print('Model has {} trainable variables and {} trainable parameters'.format(
    n_variables, n_parameters))
print()

# sys.exit()

labels_maps = list(itertools.permutations(range(n_classes)))
print('labels_maps: {}'.format(labels_maps))

gpu_memory_fraction = args.gpu_memory_fraction
allow_memory_growth = args.allow_memory_growth

session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
session_config.gpu_options.allow_growth = allow_memory_growth
session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

sess = tf.Session(config=session_config)

init = tf.global_variables_initializer()
saver_latest = tf.train.Saver(max_to_keep=1)
saver_acc = tf.train.Saver(max_to_keep=1)
saver_loss = tf.train.Saver(max_to_keep=1)

with sess.as_default():
    init.run()

if min_indices > max_indices:
    raise AssertionError('min_indices cannot be larger than min_indices')

if not train_labels_path:
    train_labels_path = os.path.join(os.path.dirname(train_images_path), 'labels')

src_files, src_labels_list, total_frames = readData(train_images_path, train_images_ext, train_labels_path,
                                                    train_labels_ext)
if start_id < 0:
    if end_id < 0:
        raise AssertionError('end_id must be non negative for random selection')
    elif end_id >= total_frames:
        raise AssertionError('end_id must be less than total_frames for random selection')
    print('Using {} random images for training'.format(end_id + 1))
    img_ids = np.random.choice(total_frames, end_id + 1, replace=False)
else:
    if end_id < start_id:
        end_id = total_frames - 1
    print('Using all {} images for training'.format(end_id - start_id + 1))
    img_ids = range(start_id, end_id + 1)

if start_id < 0:
    log_template = '{:d}_{}_{}_{}_random_{}_{}'.format(
        int(args.index_percent), min_indices, max_indices, height, end_id + 1, loss_type)
else:
    log_template = '{:d}_{}_{}_{}_{}_{}_{}'.format(
        int(args.index_percent), min_indices, max_indices, height, start_id, end_id, loss_type)

if psi_act_type > 0:
    log_template = '{}_{}'.format(log_template, model.psi_act_name)
if not log_dir:
    log_root_dir = 'log'
else:
    log_root_dir = os.path.dirname(log_dir)
    log_template = '{}_{}'.format(os.path.basename(log_dir), log_template)

if n_layers > 0:
    log_template = '{}_{}_layers'.format(log_template, n_layers)

log_dir = os.path.join(log_root_dir, log_template)

save_weights_path = os.path.join(log_dir, 'weights')
save_weights_acc_path = os.path.join(log_dir, 'weights_acc')
save_weights_loss_path = os.path.join(log_dir, 'weights_loss')

save_path = os.path.join(log_dir, 'results')

if not os.path.isdir(save_weights_path):
    os.makedirs(save_weights_path)
if not os.path.isdir(save_weights_acc_path):
    os.makedirs(save_weights_acc_path)
if not os.path.isdir(save_weights_loss_path):
    os.makedirs(save_weights_loss_path)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

log_fname = os.path.join(save_path, 'log_{:s}.txt'.format(getDateTime()))

arg_names = [a for a in dir(args) if not a.startswith('__')]
with open(log_fname, 'a') as log_fid:
    for arg in arg_names:
        log_fid.write('{}: {}\n'.format(arg, getattr(args, arg)))

# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True

# config = tf.ConfigProto(device_count={'GPU': gpu_id})

label_diff = int(255.0 / (n_classes - 1))

start_epoch = 0
min_loss = np.inf
min_loss_epoch = -1
max_pix_acc = 0
max_pix_acc_epoch = -1

if load_weights:
    if load_weights == '1':
        load_weights = save_weights_path + '/'
    elif load_weights == '0':
        load_weights = ''
    if load_weights.endswith('/'):
        ckpt_path = tf.train.latest_checkpoint(load_weights)
    else:
        ckpt_path = load_weights
    if ckpt_path:
        print('Restoring weights from {}'.format(ckpt_path))
        try:
            saver_loss.restore(sess, ckpt_path)
            try:
                start_epoch = int(ckpt_path.split('-')[-1]) + 1
            except:
                pass
            if load_log:
                last_log_file = os.path.join(save_path, load_log)
            else:
                log_file_list = [os.path.join(save_path, k) for k in os.listdir(save_path) if
                                 k.startswith('log_') and k.endswith('.txt')]

                if len(log_file_list) <= 1:
                    print('No previous log files found')
                    last_log_file = ''
                else:
                    log_file_list.sort()
                    last_log_file = log_file_list[-2]
            print_and_write('Loading previous log from {}\n'.format(last_log_file), log_fname)
            try:
                if last_log_file:
                    last_log_line = open(last_log_file, 'r').readlines()[-1]
                    last_log_data = [k.strip() for k in last_log_line.split() if k.strip()]

                    print('last_log_data:\n {}'.format(last_log_data))

                    min_loss_data = last_log_data[7].split('(')
                    max_pix_acc_data = last_log_data[11].split('(')

                    min_loss = float(min_loss_data[0])
                    min_loss_epoch = int(min_loss_data[1].replace(')', ''))

                    max_pix_acc = float(max_pix_acc_data[0])
                    mean_pix_acc = float(last_log_data[9])
                    max_pix_acc_epoch = int(max_pix_acc_data[1].replace(')', ''))

                    print_and_write('Loaded min_loss: {}({}) max_pix_acc: {}({}) mean_pix_acc: {}'.format(
                        min_loss, min_loss_epoch, max_pix_acc, max_pix_acc_epoch, mean_pix_acc), log_fname)

                    if len(last_log_data) >= 14:
                        learning_rate = float(last_log_data[13])
                        print_and_write('learning_rate: {}'.format(learning_rate))

            except BaseException as e:
                print_and_write('Loading log was unsuccessful: {}'.format(e), log_fname)
        except BaseException as e:
            print_and_write('Restoring weights was unsuccessful so training from scratch: {}'.format(e), log_fname)

train_images = []
# train_labels = []
train_indices = []
# train_Y = []

optimize_label_map = (loss_type != 4 and loss_type != 5)

if preload_images:
    print('Preloading images')
else:
    print('Not preloading images')

print('Getting training data...')
if index_ratio == 0:
    print('Using a fixed set of {} pixels per class'.format(max_indices))

_n_training_images = len(img_ids)

for _id, img_id in enumerate(img_ids):
    # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
    img_fname = src_files[img_id]
    img_fname_no_ext = os.path.splitext(img_fname)[0]

    labels_img_fname = os.path.join(train_labels_path, img_fname_no_ext + '.{}'.format(train_labels_ext))
    labels_img = imread(labels_img_fname)

    if labels_img is None:
        raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

    if len(labels_img.shape) == 3:
        labels_img = labels_img[:, :, 0].squeeze()

    # print('min: {} max: {}'.format(
    #     np.min(labels_img.flatten()),
    #     np.max(labels_img.flatten()))
    # )
    # np.savetxt('labels_img.txt', labels_img, fmt='%d')
    # sys.exit()

    labels_indices = []

    # Y = np.zeros((height * width, n_classes), dtype=np.float32)
    skip_image = 0
    for class_id in range(n_classes):
        class_indices = np.flatnonzero(labels_img == class_id)
        if class_indices.shape[0] < min_indices:
            skip_image = 1
            print('\nimg {} class {} class_indices.shape: {} '.format(img_id + 1, class_id, class_indices.shape))
            break
        # Y[class_indices, class_id] = 1

        # y_save_path = os.path.join(save_path, '{}_{}.png'.format(img_fname_no_ext, class_id))
        # imsave(y_save_path, np.reshape(Y[:, class_id]*255, (height, width)).astype(np.uint8))

        if index_ratio == 0:
            class_indices = np.random.choice(class_indices, (max_indices, 1), replace=False)

        labels_indices.append(class_indices)
    if skip_image:
        continue

    src_img_fname = os.path.join(train_images_path, img_fname)

    if preload_images:
        src_img = imread(src_img_fname)
        if src_img is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

        src_img = src_img / np.amax(src_img)
        src_img = np.reshape(src_img, (1, height, width, 3)).astype(np.float32)
        train_images.append(src_img)
    else:
        train_images.append(src_img_fname)

    # train_labels.append(labels_img)
    train_indices.append(labels_indices)
    # train_Y.append(Y)
    sys.stdout.write('\rDone {}/{} images'.format(_id + 1, _n_training_images))
    sys.stdout.flush()

print()

n_images = len(train_images)

if n_images == 0:
    raise AssertionError('no valid training images found')
# sys.exit()

print('Getting testing data...')
if test_images_path:
    if not test_labels_path:
        test_labels_path = os.path.join(os.path.dirname(test_images_path), 'labels')

    test_file_list, test_labels_list, total_test_frames = readData(test_images_path, test_images_ext,
                                                                   test_labels_path,
                                                                   test_labels_ext)
else:
    test_file_list, test_labels_list, total_test_frames = src_files, src_labels_list, total_frames
    test_images_path, test_labels_path = train_images_path, train_labels_path
    test_images_ext, test_labels_ext = train_images_ext, train_labels_ext

if test_start_id < 0:
    if test_end_id < 0:
        raise AssertionError('test_end_id must be non negative for random selection')
    elif end_id >= total_test_frames:
        raise AssertionError('test_end_id must be less than total_test_frames for random selection')
    print('Using {} random images for evaluation'.format(test_end_id + 1))
    test_img_ids = np.random.choice(total_test_frames, test_end_id + 1, replace=False)
else:
    if test_end_id < test_start_id:
        test_end_id = total_test_frames - 1
    test_img_ids = range(test_start_id, test_end_id + 1)

test_images = []
test_images_orig = []
test_labels = []
test_names = []

_n_test_images = len(test_img_ids)

for _id, img_id in enumerate(test_img_ids):

    # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
    img_fname = test_file_list[img_id]
    img_fname_no_ext = os.path.splitext(img_fname)[0]
    src_img_fname = os.path.join(test_images_path, img_fname)
    labels_img_fname = os.path.join(test_labels_path, img_fname_no_ext + '.{}'.format(test_labels_ext))

    if preload_images:
        src_img = imread(src_img_fname)
        if src_img is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))
        if save_test and save_stitched:
            test_images_orig.append(src_img)

        src_img = src_img / np.amax(src_img)
        src_img = np.reshape(src_img, (1, height, width, 3)).astype(np.float32)
        test_images.append(src_img)

        labels_img = imread(labels_img_fname)
        if labels_img is None:
            raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))
        if len(labels_img.shape) == 3:
            labels_img = labels_img[:, :, 0].squeeze()

        test_labels.append(labels_img)
    else:
        test_images.append(src_img_fname)
        test_labels.append(labels_img_fname)

    test_names.append(img_fname_no_ext)
    sys.stdout.write('\rDone {}/{} images'.format(_id + 1, _n_test_images))
    sys.stdout.flush()

print()

n_test_images = len(test_images)

# feed_dict_list = []
# for img_id in range(n_images):
#     src_img = train_images[img_id]
#     label_img = train_labels[img_id]
#
#     feed_dict = {m.X: src_img, m.lr: learning_rate}
#     for class_id in range(n_classes):
#         class_indices = np.flatnonzero(label_img == class_id)
#         feed_dict.update({m.class_indices[class_id]: class_indices})
#     feed_dict_list.append(feed_dict)

pix_acc = np.zeros((n_test_images,))
# mean_acc = np.zeros((n_test_images,))
# mean_IU = np.zeros((n_test_images,))
# fw_IU = np.zeros((n_test_images,))

feed_dict = None

sys.stdout.write('Saving latest checkpoint to: {}\n'.format(save_weights_path))
sys.stdout.write('Saving max accuracy checkpoint to: {}\n'.format(save_weights_acc_path))
sys.stdout.write('Saving min loss checkpoint to: {}\n'.format(save_weights_loss_path))
sys.stdout.write('Saving results to: {}\n'.format(save_path))
sys.stdout.write('Saving log to: {}\n'.format(log_fname))
sys.stdout.flush()

print_and_write('Training on {:d} images'.format(n_images), log_fname)
print_and_write('Evaluating on {:d} images'.format(n_test_images), log_fname)
if restore_on_nan:
    print_and_write('Using previous checkpoint restoring on NaN loss', log_fname)
else:
    print_and_write('Using remaining images skipping on NaN loss', log_fname)
print_and_write('Using {} as psi activation function'.format(model.psi_act_name))

img_ids = list(range(n_images))
epoch_id = start_epoch
while epoch_id < n_epochs:
    # print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    # losses = []
    avg_loss = 0
    # nan_loss = False
    for img_id in img_ids:
        overall_start_t = time.time()

        if preload_images:
            src_img = train_images[img_id]
        else:
            src_img_fname = train_images[img_id]
            src_img = imread(src_img_fname)
            if src_img is None:
                raise SystemError('Source image could not be read from: {}'.format(src_img_fname))
            src_img = src_img / np.amax(src_img)
            src_img = np.reshape(src_img, (1, height, width, 3)).astype(np.float32)

        # height, width, _ = src_img.shape
        # print('height: ', height)
        # print('width: ', width)

        # src_img_exp = np.expand_dims(src_img, axis=0)
        feed_dict = {model.X: src_img, model.lr: learning_rate}

        # feed_dict.update({m.height: height, m.width: width})

        labels_indices = train_indices[img_id]
        Y = np.zeros((height * width, n_classes), dtype=np.float32)
        for class_id in range(n_classes):
            if index_ratio == 0:
                class_indices = labels_indices[class_id]
            else:
                n_indices = min(max(min_indices, int(index_ratio * labels_indices[class_id].shape[0])),
                                max_indices)
                # print('n_indices: ', n_indices)

                class_indices = np.random.choice(labels_indices[class_id], (n_indices, 1), replace=False)
            Y[class_indices, class_id] = 1

            feed_dict.update({model.class_indices[class_id]: class_indices})

        feed_dict.update({model.Y: Y})
        # feed_dict = feed_dict_list[img_id]

        start_t = time.time()
        psi, phi_den, phi, loss, _ = sess.run(
            (model.psi, model.phi_den, model.phi, model.loss_convnet, model.training_op),
            feed_dict=feed_dict)
        end_t = time.time()

        fps = 1.0 / (end_t - start_t)

        nan_loss = 0
        if math.isnan(loss):
            print_and_write('\nNaN loss encountered for image {} in epoch {}'.format(img_id, epoch_id), log_fname)

            # print_and_write('phi:\n {}'.format(phi), log_fname)
            # print_and_write('psi:\n {}'.format(psi), log_fname)
            # print_and_write('phi_den:\n {}'.format(phi_den), log_fname)

            np.savetxt(os.path.join(save_path, 'phi_{}_{}.dat'.format(img_id, epoch_id)),
                       np.asarray(phi), delimiter='\t', fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'psi_{}_{}.dat'.format(img_id, epoch_id)),
                       np.asarray(psi), delimiter='\t', fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'phi_den_{}_{}.dat'.format(img_id, epoch_id)),
                       np.asarray(phi_den), delimiter='\t', fmt='%.4f')

            if restore_on_nan:
                ckpt_path = tf.train.latest_checkpoint(save_weights_path)
                print_and_write('Restoring weights from {}'.format(ckpt_path), log_fname)
                saver_loss.restore(sess, ckpt_path)

                img_ids.remove(img_id)
                n_images -= 1
                nan_loss = 1
                break
            else:
                continue

        # losses.append(loss)
        avg_loss += (loss - avg_loss) / (img_id + 1)

        overall_end_t = time.time()
        overall_fps = 1.0 / (overall_end_t - overall_start_t)

        sys.stdout.write('\rDone {:5d}/{:5d} frames in epoch {:5d} ({:6.2f}/{:6.2f} fps) avg_loss: {:f}'.format(
            img_id + 1, n_images, epoch_id, fps, overall_fps, avg_loss))
        sys.stdout.flush()

        # if save_after_each_step:
        #     loss_convnet_val = m.loss_convnet.eval(feed_dict=feed_dict)
        #     print("Image:", img_id + 1, "Convnet loss:", loss_convnet_val)

        # Seg = np.zeros((height, width))
        # for i in range(height):
        #     for j in range(width):
        #         val = -1
        #         label = -1
        #         for n in range(n_classes):
        #             if ClassIndicator[i, j, n] > val:
        #                 val = ClassIndicator[i, j, n]
        #                 label = n
        #         Seg[i, j] = label*label_diff

    if nan_loss:
        continue

    print()

    if epoch_id % eval_every == 0:
        optimal_label_map = None
        if optimize_label_map:
            print('\nGetting optimal label map...')
            random_img_ids = np.random.choice(n_test_images, 10, replace=False)
            _max_pix_acc = 0
            for labels_map in labels_maps:
                _mean_pix_acc = 0
                random_id = 0
                for _img_id in random_img_ids:
                    if preload_images:
                        test_img = test_images[_img_id]
                        if save_test:
                            test_img_orig = test_images_orig[_img_id]
                        labels_img = test_labels[_img_id]
                    else:
                        src_img_fname = test_images[_img_id]
                        test_img_orig = imread(src_img_fname)
                        if test_img_orig is None:
                            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))
                        test_img = test_img_orig / np.amax(test_img_orig)
                        test_img = np.reshape(test_img, (1, height, width, 3)).astype(np.float32)

                        labels_img_fname = test_labels[_img_id]
                        labels_img = imread(labels_img_fname)
                        if labels_img is None:
                            raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))
                        if len(labels_img.shape) == 3:
                            labels_img = labels_img[:, :, 0].squeeze()

                    phi_val = model.phi.eval(session=sess, feed_dict={model.X: test_img})
                    ClassIndicator = phi_val.reshape((height, width, n_classes))
                    labels = np.argmax(ClassIndicator, axis=2)

                    labels_mapped = np.vectorize(lambda x: labels_map[x])(labels)
                    _pix_acc = eval.pixel_accuracy(labels_mapped, labels_img)
                    random_id += 1
                    _mean_pix_acc += (_pix_acc - _mean_pix_acc) / random_id

                if _mean_pix_acc > _max_pix_acc:
                    _max_pix_acc = _mean_pix_acc
                    optimal_label_map = labels_map

            print('optimal_label_map: {}'.format(optimal_label_map))
            print('_max_pix_acc: {}'.format(_max_pix_acc))

        print('\nTesting...')
        mean_pix_acc = 0
        for img_id in range(n_test_images):
            overall_start_t = time.time()
            if preload_images:
                test_img = test_images[img_id]
                if save_test:
                    test_img_orig = test_images_orig[img_id]
                labels_img = test_labels[img_id]
            else:
                src_img_fname = test_images[img_id]
                test_img_orig = imread(src_img_fname)
                if test_img_orig is None:
                    raise SystemError('Source image could not be read from: {}'.format(src_img_fname))
                test_img = test_img_orig / np.amax(test_img_orig)
                test_img = np.reshape(test_img, (1, height, width, 3)).astype(np.float32)

                labels_img_fname = test_labels[img_id]
                labels_img = imread(labels_img_fname)
                if labels_img is None:
                    raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))
                if len(labels_img.shape) == 3:
                    labels_img = labels_img[:, :, 0].squeeze()

            start_t = time.time()

            phi_val = model.phi.eval(session=sess, feed_dict={model.X: test_img})
            ClassIndicator = phi_val.reshape((height, width, n_classes))
            labels = np.argmax(ClassIndicator, axis=2)

            end_t = time.time()

            fps = 1.0 / (end_t - start_t)
            fps_with_input = 1.0 / (end_t - overall_start_t)

            if optimize_label_map:
                labels = np.vectorize(lambda x: optimal_label_map[x])(labels)

            pix_acc[img_id] = eval.pixel_accuracy(labels, labels_img)
            # mean_acc[img_id] = eval.mean_accuracy(labels, labels_img)
            # mean_IU[img_id] = eval.mean_IU(labels, labels_img)
            # fw_IU[img_id] = eval.frequency_weighted_IU(labels, labels_img)

            if save_test:
                print('Saving test result')
                Seg = (labels * label_diff).astype(np.uint8)
                seg_save_path = os.path.join(save_path, '{:s}_epoch_{:d}.png'.format(
                    test_names[img_id], epoch_id + 1))

                if save_stitched:
                    gt_seq = (labels_img * label_diff).astype(np.uint8)
                    if len(gt_seq.shape) != 3:
                        gt_seq = np.stack((gt_seq, gt_seq, gt_seq), axis=2)
                    if len(Seg.shape) != 3:
                        Seg = np.stack((Seg, Seg, Seg), axis=2)
                    Seg = np.concatenate((test_img_orig, gt_seq, Seg), axis=1)
                imsave(seg_save_path, Seg)

            overall_end_t = time.time()
            overall_fps = 1.0 / (overall_end_t - overall_start_t)
            mean_pix_acc += (pix_acc[img_id] - mean_pix_acc) / (img_id + 1)

            sys.stdout.write('\rDone {:5d}/{:5d} frames in epoch {:5d} ({:6.2f}({:6.2f}, {:6.2f}) fps) '
                             'pix_acc: {:.10f}'.format(
                img_id + 1, n_test_images, epoch_id, fps, fps_with_input, overall_fps, mean_pix_acc))
            sys.stdout.flush()

        print()

        # mean_pix_acc = np.mean(pix_acc)
        if mean_pix_acc > max_pix_acc:
            max_pix_acc = mean_pix_acc
            max_pix_acc_epoch = epoch_id
            saver_acc.save(sess, os.path.join(save_weights_acc_path, 'model.ckpt-{}'.format(epoch_id)))

    # loss_convnet_val = m.loss_convnet.eval(feed_dict=feed_dict)
    # loss_convnet_val = np.mean(losses)
    loss_convnet_val = avg_loss
    if loss_convnet_val < min_loss:
        min_loss = loss_convnet_val
        min_loss_epoch = epoch_id
        saver_loss.save(sess, os.path.join(save_weights_loss_path, 'model.ckpt-{}'.format(epoch_id)))

    saver_latest.save(sess, os.path.join(save_weights_path, 'model.ckpt-{}'.format(epoch_id)))

    print_and_write("{:s} :: epoch: {:4d} loss: {:.10f} min_loss: {:.10f}({:d}) pix_acc: {:.10f} " \
                    "max_pix_acc:  {:.10f}({:d}) lr: {:.10f}".format(
        log_template, epoch_id, loss_convnet_val, min_loss, min_loss_epoch, mean_pix_acc,
        max_pix_acc, max_pix_acc_epoch, learning_rate), log_fname)

    epoch_id += 1

    if lr_dec_epochs > 0 and (epoch_id + 1) % lr_dec_epochs == 0:
        learning_rate = lr_dec_rate * learning_rate
