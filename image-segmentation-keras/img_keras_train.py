import argparse, os, sys, time, glob, math
import Models, LoadBatches
import numpy as np
from keras import backend as K
from keras import losses, layers, metrics

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--train_images", type=str)
parser.add_argument("--train_annotations", type=str)
parser.add_argument("--train_batch_size", type=int, default=2)

parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)

parser.add_argument('--validate', action='store_false')
parser.add_argument("--val_images", type=str, default="")
parser.add_argument("--val_annotations", type=str, default="")
parser.add_argument("--val_batch_size", type=int, default=2)

parser.add_argument("--steps_per_epoch", type=int, default=0)

parser.add_argument("--selective_loss", type=int, default=0)

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--load_weights", type=str, default="")

parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--optimizer_name", type=str, default="adadelta")
parser.add_argument("--load_from_log", type=str, default="")

parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=-1)


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.train_batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
steps_per_epoch = args.steps_per_epoch
load_from_log = args.load_from_log
selective_loss = args.selective_loss
optimizer_name = args.optimizer_name
model_name = args.model_name
start_id = args.start_id
end_id = args.end_id


class LogWriter:
    def __init__(self, fname):
        self.fname=fname

    def write(self, _str):
        print(_str + '\n')
        with open(self.fname, 'a') as fid:
            fid.write(_str + '\n')


def delete_weights_and_model(ep, postfix):
    weights_path = os.path.join(save_weights_path, "weights_{:s}_{:d}.h5".format(postfix, ep + 1))
    if os.path.exists(weights_path):
        os.remove(weights_path)
    models_path = os.path.join(save_weights_path, "model_{:s}_{:d}.h5".format(postfix, ep + 1))
    if os.path.exists(models_path):
        os.remove(models_path)


def save_weights_and_model(m, ep, postfix):
    m.save_weights(os.path.join(save_weights_path, "weights_{:s}_{:d}.h5".format(postfix, ep + 1)))
    m.save(os.path.join(save_weights_path, "model_{:s}_{:d}.h5".format(postfix, ep + 1)))


def printStatus(status_list, var_name):
    sys.stdout.write('{:s}: {:f} in epoch {:d}\t'.format(
        var_name, status_list[0], status_list[1]))
    acc = float(status_list[2]['acc'][0])
    loss = float(status_list[2]['loss'][0])
    val_acc = float(status_list[2]['val_acc'][0])
    val_loss = float(status_list[2]['val_loss'][0])
    sys.stdout.write('acc: {:.4f}\t'.format(acc))
    sys.stdout.write('val_acc: {:.4f}\t'.format(val_acc))
    sys.stdout.write('loss: {:.4f}\t'.format(loss))
    sys.stdout.write('val_loss: {:.4f}\n'.format(val_loss))
    sys.stdout.flush()


def getDateTime():
    return time.strftime("%y%m%d_%H%M", time.localtime())


import theano.tensor as T


def selective_categorical_crossentropy(y_true, y_pred):
    writer.write('orig y_true: {}'.format(y_true))
    writer.write('orig y_pred: {}'.format(y_pred))

    # y_true_col = y_true[:, 0]

    mask = K.greater_equal(y_true, K.zeros_like(y_true))  # boolean tensor, mask[i] = True iff x[i] > 1
    # boolean_mask = layers.Lambda(lambda y: tf.boolean_mask(y, mask))

    writer.write('orig mask: {}'.format(mask))

    # y_true2 = layers.Lambda(lambda y_true: y_true * mask, output_shape=y_true._keras_shape)
    # y_pred2 = layers.Lambda(lambda y_pred: y_pred * mask, output_shape=y_pred._keras_shape)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    writer.write('y_true2: {}'.format(y_true))
    writer.write('y_pred2: {}'.format(y_pred))

    #
    # print('y_true: {}'.format(y_true))
    # print('y_pred: {}'.format(y_pred))
    #
    # print('orig y_true.shape: {}'.format(np.array(y_true).shape))
    # print('orig y_pred.shape: {}'.format(np.array(y_pred).shape))

    # print('mask.shape: {}'.format(mask._keras_shape))

    # print('orig y_pred.shape: {}'.format(y_pred._keras_shape))
    # print('orig y_true.shape: {}'.format(y_true._keras_shape))

    # print('y_pred2.shape: {}'.format(y_pred2._keras_shape))
    # print('y_true2.shape: {}'.format(y_true2._keras_shape))

    # valid_idx = np.array(y_true[:, 0] != -1)
    # print('valid_idx.shape: {}'.format(valid_idx.shape))
    #
    # y_true = y_true[valid_idx, :]
    # y_pred = y_pred[valid_idx, :]
    # print('y_true.shape: {}'.format(y_true.shape))
    # print('y_pred.shape: {}'.format(y_pred.shape))

    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping

    y_pred = T.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return T.nnet.categorical_crossentropy(y_pred, y_true)

    # return K.categorical_crossentropy(y_pred, y_true)


if not os.path.isdir(save_weights_path):
    os.makedirs(save_weights_path)

log_fname = os.path.join(save_weights_path, 'log_{:s}.txt'.format(getDateTime()))
writer = LogWriter(log_fname)

writer.write('Saving weights and model to: {}'.format(save_weights_path))
writer.write('Saving log to: {}'.format(log_fname))

arg_names = [a for a in dir(args) if not a.startswith('__')]
with open(log_fname, 'a') as log_fid:
    for arg in arg_names:
        log_fid.write('{}: {}\n'.format(arg, getattr(args, arg)))

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32,
            'deeplab': Models.Deeplabv3.Deeplabv3}
modelFN = modelFns[model_name]

if selective_loss:
    writer.write('Using selective loss {}'.format(selective_loss))
    # loss_fn = losses.categorical_crossentropy
    loss_fn = selective_categorical_crossentropy
else:
    loss_fn = losses.categorical_crossentropy

model = modelFN(n_classes, input_height=input_height, input_width=input_width)
model.compile(loss=loss_fn,
              optimizer=optimizer_name,
              metrics=['accuracy'])

if model_name == 'deeplab':
    ordering = 'channels_last'
else:
    ordering = 'channels_first'

start_epoch_id = 0

weight_types = ['min_loss', 'min_val_loss', 'min_mean_loss', 'max_acc', 'max_val_acc', 'max_mean_acc']
weight_stats = {
    'min_loss': [np.inf, 0, None],
    'min_val_loss': [np.inf, 0, None],
    'min_mean_loss': [np.inf, 0, None],
    'max_acc': [-np.inf, 0, None],
    'max_val_acc': [-np.inf, 0, None],
    'max_mean_acc': [-np.inf, 0, None],
}
is_better = {}
for weight_type in weight_types:
    if weight_type.startswith('max_'):
        is_better[weight_type] = lambda x, y: x > y
    else:
        is_better[weight_type] = lambda x, y: x < y

if load_weights == '0':
    load_weights = ''

if load_weights:
    if not os.path.isfile(load_weights):
        if load_weights == '1':
            load_weights = 'latest'
        weight_files = [f for f in os.listdir(save_weights_path) if load_weights in f and f.endswith('.h5')]
        if not weight_files:
            writer.write('No weight files found matching {} in {}'.format(load_weights, save_weights_path))
            load_weights = ''
        else:
            weight_files.sort()
            load_weights = os.path.join(save_weights_path, weight_files[-1])

if load_weights:
    load_weights_fname = os.path.splitext(os.path.basename(load_weights))[0]
    start_epoch_id = int(load_weights_fname.split('_')[-1]) - 1

    writer.write('Loading weights from: {} with epoch'.format(load_weights, start_epoch_id + 1))
    model.load_weights(load_weights)

    log_files = [f for f in os.listdir(save_weights_path) if
                 f.startswith('log_') and f.endswith('.txt') and f != os.path.basename(log_fname)]
    log_files.sort()
    curr_log_id = -1

    # writer.write('Restoring log info from:')
    if not os.path.isfile(load_from_log):
        load_from_log = os.path.join(save_weights_path, log_files[-1])
        # all_log_lines = []
        # for _file in log_files:
        #     _file_path = os.path.join(save_weights_path, _file)
        #     writer.write('{}'.format(_file_path))
        #     all_log_lines += open(_file_path, 'r').readlines()
    # else:
    #     writer.write('{}'.format(load_from_log))
    #     all_log_lines = open(load_from_log, 'r').readlines()

    writer.write('Restoring log info from: {}'.format(load_from_log))
    all_log_lines = open(load_from_log, 'r').readlines()
    weight_id = 0
    n_weight_types = len(weight_types)

    while weight_id < n_weight_types:
        weight_type = weight_types[weight_id]
        last_log_data = last_log_line = None
        try:
            log_lines = [k.strip() for k in all_log_lines if k.startswith(weight_type)]
            last_log_line = log_lines[-1]
            last_log_data = last_log_line.split()
            _val = float(last_log_data[1][1:-1])
            _epoch = int(last_log_data[2][:-1])
            weight_stats[weight_type][0] = _val
            weight_stats[weight_type][1] = _epoch
            weight_stats[weight_type][2] = {
                'acc': [float(last_log_data[4][1:-2])],
                'loss': [float(last_log_data[6][1:-2])],
                'val_acc': [float(last_log_data[8][1:-2])],
                'val_loss': [float(last_log_data[10][1:-3])],
            }
            writer.write('\n' + last_log_line)
        except BaseException as e:
            writer.write('weight_type: {}'.format(weight_type))
            if last_log_line is not None:
                writer.write('last_log_line: {}'.format(last_log_line))
            if last_log_data is not None:
                writer.write('last_log_data: {}'.format(last_log_data))
            writer.write('Error in parsing log file: {}'.format(e))
            sys.exit(1)
            # try:
            #     curr_log_id -= 1
            #     load_from_log = os.path.join(save_weights_path, log_files[curr_log_id])
            #     writer.write('Trying to restore log from {}'.format(load_from_log))
            #     continue
            # except:
            #     sys.exit(1)
        weight_id += 1
    print('weight_stats: {}'.format(weight_stats))

writer.write("Model output shape: {}".format(model.output_shape))

trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

writer.write('Total params: {:,}'.format(trainable_count + non_trainable_count))
writer.write('Trainable params: {:,}'.format(trainable_count))
writer.write('Non-trainable params: {:,}'.format(non_trainable_count))

# sys.exit()

output_height = model.outputHeight
output_width = model.outputWidth

images = glob.glob(train_images_path + "*.jpg") + glob.glob(train_images_path + "*.png") + glob.glob(
    train_images_path + "*.jpeg")
segmentations = glob.glob(train_segs_path + "*.jpg") + glob.glob(train_segs_path + "*.png") + glob.glob(
    train_segs_path + "*.jpeg")

assert len(images) == len(segmentations)

if end_id < start_id:
    end_id = len(images) - 1

n_train_images = end_id - start_id + 1

if steps_per_epoch <= 0:
    steps_per_epoch = int(math.ceil(n_train_images / float(train_batch_size)))

writer.write('Training on {:d} images with batch size {:d} and {:d} steps per epoch'.format(
    n_train_images, train_batch_size, steps_per_epoch))

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width, selective_loss,
                                           ordering, start_id, end_id, debug_mode=1)

train_args = {'epochs': 1}

if validate:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations
    val_batch_size = args.val_batch_size

    val_images = glob.glob(val_images_path + "*.jpg") + glob.glob(val_images_path + "*.png") + glob.glob(
        val_images_path + "*.jpeg")

    n_val_images = len(val_images)

    validation_steps = int(math.ceil(n_val_images / float(train_batch_size)))
    writer.write('Validating on {:d} images with batch size {:d} and {:d} steps'.format(
        n_val_images, val_batch_size, validation_steps))

    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width, ordering=ordering)
    train_args.update({'validation_data': G2, 'validation_steps': validation_steps})

for weight_type in weight_types:
    _stat = weight_stats[weight_type]
    with open(log_fname, 'a') as log_fid:
        log_fid.write('{}: {}\n'.format(weight_type, _stat))

for ep in range(start_epoch_id, epochs):
    print('Epoch {}/{}'.format(ep + 1, epochs))

    history = model.fit_generator(G, steps_per_epoch, **train_args)

    acc = float(history.history['acc'][0])
    loss = float(history.history['loss'][0])
    val_acc = float(history.history['val_acc'][0])
    val_loss = float(history.history['val_loss'][0])
    mean_acc = (acc + val_acc) / 2.0
    mean_loss = (loss + val_loss) / 2.0

    curr_weight_stats = {
        'max_acc': acc,
        'max_val_acc': val_acc,
        'max_mean_acc': mean_acc,
        'min_loss': loss,
        'min_val_loss': val_loss,
        'min_mean_loss': mean_loss,
    }

    delete_weights_and_model(ep - 1, 'latest')
    save_weights_and_model(model, ep, 'latest')

    for weight_type in weight_types:
        _stat = weight_stats[weight_type]
        if is_better[weight_type](curr_weight_stats[weight_type], _stat[0]):
            delete_weights_and_model(_stat[1] - 1, weight_type)
            _stat[0] = curr_weight_stats[weight_type]
            _stat[1] = ep + 1
            _stat[2] = history.history
            save_weights_and_model(model, ep, weight_type)
            with open(log_fname, 'a') as log_fid:
                log_fid.write('{}: {}\n'.format(weight_type, _stat))
        printStatus(_stat, weight_type)

    # if val_acc > max_val_acc[0]:
    #     delete_weights_and_model(max_val_acc[1] - 1, 'max_val_acc')
    #     max_val_acc[0] = val_acc
    #     max_val_acc[1] = ep + 1
    #     max_val_acc[2] = history.history
    #     save_weights_and_model(m, ep, 'max_val_acc')
    #     with open(log_fname, 'a') as log_fid:
    #         log_fid.write('max_val_acc: {}\n'.format(max_val_acc))
    #
    # if mean_acc > max_mean_acc[0]:
    #     delete_weights_and_model(max_mean_acc[1] - 1, 'max_mean_acc')
    #     max_mean_acc[0] = mean_acc
    #     max_mean_acc[1] = ep + 1
    #     max_mean_acc[2] = history.history
    #     save_weights_and_model(m, ep, 'max_mean_acc')
    #     with open(log_fname, 'a') as log_fid:
    #         log_fid.write('max_mean_acc: {}\n'.format(max_mean_acc))
    #
    # if loss < min_loss[0]:
    #     delete_weights_and_model(min_loss[1] - 1, 'min_loss')
    #     min_loss[0] = loss
    #     min_loss[1] = ep + 1
    #     min_loss[2] = history.history
    #     save_weights_and_model(m, ep, 'min_loss')
    #     with open(log_fname, 'a') as log_fid:
    #         log_fid.write('min_loss: {}\n'.format(min_loss))
    #
    # if val_loss < min_val_loss[0]:
    #     delete_weights_and_model(min_val_loss[1] - 1, 'min_val_loss')
    #     min_val_loss[0] = val_loss
    #     min_val_loss[1] = ep + 1
    #     min_val_loss[2] = history.history
    #     save_weights_and_model(m, ep, 'min_val_loss')
    #     with open(log_fname, 'a') as log_fid:
    #         log_fid.write('min_val_loss: {}\n'.format(min_val_loss))
    #
    # if mean_loss < min_mean_loss[0]:
    #     delete_weights_and_model(min_mean_loss[1] - 1, 'min_mean_loss')
    #     min_mean_loss[0] = mean_loss
    #     min_mean_loss[1] = ep + 1
    #     min_mean_loss[2] = history.history
    #     save_weights_and_model(m, ep, 'min_mean_loss')
    #     with open(log_fname, 'a') as log_fid:
    #         log_fid.write('min_mean_loss: {}\n'.format(min_mean_loss))
    #
    # printStatus(max_acc, 'max_acc')
    # printStatus(max_val_acc, 'max_val_acc')
    # printStatus(max_mean_acc, 'max_mean_acc')
    # printStatus(min_loss, 'min_loss')
    # printStatus(min_val_loss, 'min_val_loss')
    # printStatus(min_mean_loss, 'min_mean_loss')
