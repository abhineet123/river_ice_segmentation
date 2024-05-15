import argparse
from DenseNet import DenseNet
from DenseNet2 import DenseNet2
from utils import readData, print_and_write, getDateTime
import os, sys
import numpy as np
import tensorflow as tf
# from scipy.misc.pilutil import imread, imsave
import evaluation.eval_segm as eval
import cv2
from scipy.misc.pilutil import imread, imsave
import time

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str)
parser.add_argument("--images_path", type=str)
parser.add_argument("--images_ext", type=str, default="png")
parser.add_argument("--out_ext", type=str, default="png")
parser.add_argument("--labels_path", type=str, default="")
parser.add_argument("--labels_ext", type=str, default='png')
parser.add_argument("--save_path", type=str, default="")
parser.add_argument("--height", type=int, default=224)
parser.add_argument("--width", type=int, default=224)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=-1)
parser.add_argument("--save_seg", type=int, default=1)
parser.add_argument("--save_stitched", type=int, default=1)
parser.add_argument("--allow_memory_growth", type=int, default=1)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--loss_type", type=int, default=0)
parser.add_argument("--stitch_labels", type=int, default=1)
parser.add_argument("--show_img", type=int, default=0)
parser.add_argument("--save_raw", type=int, default=1)
parser.add_argument("--normalize_labels", type=int, default=1)
parser.add_argument("--psi_act_type", type=int, default=0)
parser.add_argument("--n_layers", type=int, default=0)

args = parser.parse_args()

n_classes = args.n_classes
images_path = args.images_path
images_ext = args.images_ext
out_ext = args.out_ext
labels_path = args.labels_path
labels_ext = args.labels_ext
allow_memory_growth = args.allow_memory_growth

width = args.width
height = args.height
weights_path = args.weights_path
save_path = args.save_path
start_id = args.start_id
end_id = args.end_id
save_seg = args.save_seg
save_stitched = args.save_stitched
gpu_id = args.gpu_id
loss_type = args.loss_type

stitch_labels = args.stitch_labels
show_img = args.show_img
save_raw = args.save_raw

psi_act_type = args.psi_act_type

normalize_labels = args.normalize_labels
n_layers = args.n_layers

src_files, src_labels_list, total_frames = readData(images_path, images_ext)

if end_id < start_id:
    end_id = total_frames - 1

eval_mode = False
if labels_path and labels_ext:
    _, labels_list, labels_total_frames = readData(labels_path=labels_path, labels_ext=labels_ext)
    if labels_total_frames != total_frames:
        raise SystemError('Mismatch between no. of frames in GT and seg labels')
    eval_mode = True
else:
    save_seg = True

colors = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]

label_diff = int(255.0 / (n_classes - 1))

tf.reset_default_graph()

if n_layers == 0:
    model = DenseNet(height, width, ch=3, nclass=n_classes, loss_type=loss_type, psi_act_type=psi_act_type)
else:
    model = DenseNet2(n_layers, height, width, ch=3, nclass=n_classes, loss_type=loss_type, psi_act_type=psi_act_type)

config = tf.ConfigProto(device_count={'GPU': gpu_id})
config.gpu_options.allow_growth = allow_memory_growth

if not save_seg:
    save_raw = save_stitched = False

classes = tuple(range(n_classes))

with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    init.run()

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log_fname = os.path.join(save_path, 'log_{:s}.txt'.format(getDateTime()))

    if save_seg:
        print_and_write('Saving segmentation results to: {}'.format(save_path), log_fname)
    else:
        print_and_write('Not saving segmentation results', log_fname)

    print_and_write('Saving log to: {}'.format(log_fname), log_fname)

    save_path_raw = os.path.join(save_path, 'raw')
    if save_raw:
        print_and_write('Saving raw labels to: {}\n'.format(save_path_raw))
        if not os.path.isdir(save_path_raw):
            os.makedirs(save_path_raw)

    if weights_path.endswith('/'):
        ckpt_path = tf.train.latest_checkpoint(weights_path)
    else:
        ckpt_path = weights_path

    print_and_write('Loading weights from {}'.format(ckpt_path), log_fname)

    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(ckpt_path + '.meta')

    saver.restore(sess, ckpt_path)

    # model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print('model_vars: {}'.format(model_vars))
    #
    # graph = tf.get_default_graph()
    # print('graph: {}'.format(graph))
    #
    # sys.exit(0)

    n_frames = end_id - start_id + 1

    pix_acc = np.zeros((n_frames,))
    mean_acc = np.zeros((n_frames,))
    mean_IU = np.zeros((n_frames,))
    fw_IU = np.zeros((n_frames,))

    avg_pix_acc = avg_mean_acc = avg_mean_IU = avg_fw_IU = 0
    
    avg_mean_acc_ice = avg_mean_acc_ice_1 = avg_mean_acc_ice_2 = 0
    avg_mean_IU_ice = avg_mean_IU_ice_1 = avg_mean_IU_ice_2 = 0
    
    skip_mean_acc_ice_1 = skip_mean_acc_ice_2 = 0
    skip_mean_IU_ice_1 = skip_mean_IU_ice_2 = 0

    print_and_write('Predicting on {:d} images'.format(n_frames), log_fname)
    print_and_write('Using {} as psi activation function'.format(model.psi_act_name))

    print_diff = int(n_frames * 0.1)

    for img_id in range(start_id, end_id + 1):
        img_fname = src_files[img_id]
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        src_img_fname = os.path.join(images_path, img_fname)
        src_img_orig = imread(src_img_fname)
        if src_img_orig is None:
            raise SystemError('Source image could not be read from: {}'.format(src_img_fname))
        # print('src_img_orig.shape:', src_img_orig.shape)

        src_img = src_img_orig / np.amax(src_img_orig)
        src_img = np.reshape(src_img, (1, height, width, 3)).astype(np.float32)

        _start_t = time.time()
        phi_val = model.phi.eval(feed_dict={model.X: src_img})
        _end_t = time.time()

        fps = 1.0 / float(_end_t - _start_t)

        ClassIndicator = phi_val.reshape((height, width, n_classes))
        labels = np.argmax(ClassIndicator, axis=2)

        if save_raw:
            raw_seg_save_path = os.path.join(save_path_raw, img_fname)
            cv2.imwrite(raw_seg_save_path, labels.astype(np.uint8))

        stitched = src_img_orig
        if eval_mode:
            labels_img_fname = os.path.join(labels_path, img_fname_no_ext + '.{}'.format(labels_ext))
            labels_img_orig = imread(labels_img_fname)

            if labels_img_orig is None:
                raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

            if len(labels_img_orig.shape) == 3:
                labels_img_orig = np.squeeze(labels_img_orig[:, :, 0])

            pix_acc[img_id] = eval.pixel_accuracy(labels, labels_img_orig)
            _acc, mean_acc[img_id] = eval.mean_accuracy(labels, labels_img_orig, return_acc=1)
            _IU, mean_IU[img_id] = eval.mean_IU(labels, labels_img_orig, return_iu=1)
            fw_IU[img_id] = eval.frequency_weighted_IU(labels, labels_img_orig)

            if normalize_labels:
                labels_img = (labels_img_orig * label_diff).astype(np.uint8)
            else:
                labels_img = np.copy(labels_img_orig)

            if len(labels_img.shape) != 3:
                labels_img = np.stack((labels_img, labels_img, labels_img), axis=2)

            if save_stitched and stitch_labels:
                stitched = np.concatenate((stitched, labels_img), axis=1)
            if show_img:
                cv2.imshow('seg_img', labels_img)

        seg_save_path = os.path.join(save_path, '{}.{}'.format(img_fname_no_ext, out_ext))
        Seg = (labels * label_diff).astype(np.uint8)
        if save_stitched:
            Seg = np.stack((Seg, Seg, Seg), axis=2)
            # src_img = np.reshape(src_img, (height, width, 3))
            # print('Seg.shape:', Seg.shape)
            # print('src_img_orig.shape:', src_img_orig.shape)

            stitched = np.concatenate((stitched, Seg), axis=1)
        else:
            stitched = Seg

        if save_seg:
            cv2.imwrite(seg_save_path, stitched)

        overall_end_t = time.time()
        overall_fps = 1.0 / float(overall_end_t - _start_t)

        log_txt = 'Done {:d}/{:d} frames fps: {:.4f} ({:.4f})'.format(img_id - start_id + 1, n_frames, fps, overall_fps)
        if eval_mode:
            avg_pix_acc += (pix_acc[img_id] - avg_pix_acc) / (img_id + 1)
            avg_mean_acc += (mean_acc[img_id] - avg_mean_acc) / (img_id + 1)
            avg_mean_IU += (mean_IU[img_id] - avg_mean_IU) / (img_id + 1)
            avg_fw_IU += (fw_IU[img_id] - avg_fw_IU) / (img_id + 1)


            mean_acc_ice = np.mean(list(_acc.values())[1:])
            avg_mean_acc_ice += (mean_acc_ice - avg_mean_acc_ice) / (img_id + 1)
            try:
                mean_acc_ice_1 = _acc[1]
                avg_mean_acc_ice_1 += (mean_acc_ice_1 - avg_mean_acc_ice_1) / (img_id - skip_mean_acc_ice_1 + 1)
            except KeyError:
                print('\nskip_mean_acc_ice_1: {}'.format(img_id))
                skip_mean_acc_ice_1 += 1
            try:
                mean_acc_ice_2 = _acc[2]
                avg_mean_acc_ice_2 += (mean_acc_ice_2 - avg_mean_acc_ice_2) / (img_id - skip_mean_acc_ice_2 + 1)
            except KeyError:
                print('\nskip_mean_acc_ice_2: {}'.format(img_id))
                skip_mean_acc_ice_2 += 1

            mean_IU_ice = np.mean(list(_IU.values())[1:])
            avg_mean_IU_ice += (mean_IU_ice - avg_mean_IU_ice) / (img_id + 1)
            try:
                mean_IU_ice_1 = _IU[1]
                avg_mean_IU_ice_1 += (mean_IU_ice_1 - avg_mean_IU_ice_1) / (img_id - skip_mean_IU_ice_1 + 1)
            except KeyError:
                print('\nskip_mean_IU_ice_1: {}'.format(img_id))
                skip_mean_IU_ice_1 += 1
            try:
                mean_IU_ice_2 = _IU[2]
                avg_mean_IU_ice_2 += (mean_IU_ice_2 - avg_mean_IU_ice_2) / (img_id - skip_mean_IU_ice_2 + 1)
            except KeyError:
                print('\nskip_mean_IU_ice_2: {}'.format(img_id))
                skip_mean_IU_ice_2 += 1
                
            log_txt = "{:s} pix_acc: {:.5f} mean_acc: {:.5f} mean_IU: {:.5f} fw_IU: {:.5f}" \
                      " avg_acc_ice: {:.5f} avg_acc_ice_1: {:.5f} avg_acc_ice_2: {:.5f}" \
                      " avg_IU_ice: {:.5f} avg_IU_ice_1: {:.5f} avg_IU_ice_2: {:.5f}".format(
                log_txt,
                avg_pix_acc, avg_mean_acc, avg_mean_IU, avg_fw_IU,
                avg_mean_acc_ice, avg_mean_acc_ice_1, avg_mean_acc_ice_2,
                avg_mean_IU_ice, avg_mean_IU_ice_1, avg_mean_IU_ice_2,
            )
        sys.stdout.write('\r' + log_txt)
        sys.stdout.flush()

        if (img_id - start_id) % print_diff == 0:
            open(log_fname, 'a').write(log_txt + '\n')

sys.stdout.write('\n')
sys.stdout.flush()

if eval_mode:
    log_txt = "pix_acc\t mean_acc\t mean_IU\t fw_IU\n{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(
        avg_pix_acc, avg_mean_acc, avg_mean_IU, avg_fw_IU)
    log_txt += "mean_acc_ice\t mean_acc_ice_1\t mean_acc_ice_2\n{:.5f}\t{:.5f}\t{:.5f}\n".format(
        avg_mean_acc_ice, avg_mean_acc_ice_1, avg_mean_acc_ice_2)
    log_txt += "mean_IU_ice\t mean_IU_ice_1\t mean_IU_ice_2\n{:.5f}\t{:.5f}\t{:.5f}\n".format(
        avg_mean_IU_ice, avg_mean_IU_ice_1, avg_mean_IU_ice_2)
    print_and_write(log_txt, log_fname)
