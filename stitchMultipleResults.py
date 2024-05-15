import argparse, os, sys
import numpy as np
from densenet.utils import read_data
from scipy.misc.pilutil import imread, imsave
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)

parser.add_argument("--seg_root_dir", type=str, default='H:/UofA/617/Project/presentation')
parser.add_argument("--seg_ext", type=str, default='png')

parser.add_argument("--images_path", type=str, default='N:/Datasets/617/images/validation/images')
parser.add_argument("--images_ext", type=str, default='tif')

parser.add_argument("--labels_path", type=str, default='')
parser.add_argument("--labels_ext", type=str, default='png')

parser.add_argument("--save_path", type=str, default='H:/UofA/617/Project/presentation/stitched')

parser.add_argument("--n_classes", type=int, default=3)

parser.add_argument("--save_stitched", type=int, default=1)

parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=20)

parser.add_argument("--show_img", type=int, default=1)

parser.add_argument("--normalize_labels", type=int, default=1)
parser.add_argument("--resize_factor", type=float, default=1)

seg_dirs = [
    "vgg_unet2_max_val_acc_validation_0_20_640_640_640_640_stitched_grs_201804262122",
    "vgg_segnet_max_val_acc_validation_0_20_640_640_640_640_stitched_hml_201804262125",
    "xception_0_49_stitched_validation_0_20_640_640_640_640_grs_201804240849",
    "50_10000_10000_800_random_200_4_predict_validation_0_563_800_800_800_800_stitched_0_20_grs_201804240955",
]

n_seg_dirs = len(seg_dirs)

args = parser.parse_args()

seg_root_dir = args.seg_root_dir
seg_ext = args.seg_ext

images_path = args.images_path
images_ext = args.images_ext

labels_path = args.labels_path
labels_ext = args.labels_ext

save_path = args.save_path

n_classes = args.n_classes

end_id = args.end_id
start_id = args.start_id

show_img = args.show_img
save_stitched = args.save_stitched

normalize_labels = args.normalize_labels
resize_factor = args.resize_factor

src_files, src_labels_list, total_frames = read_data(images_path, images_ext, labels_path,
                                                     labels_ext)
if end_id < start_id:
    end_id = total_frames - 1


label_diff = int(255.0 / (n_classes - 1))

n_frames = end_id - start_id + 1

if not os.path.isdir(save_path):
    os.makedirs(save_path)
print('Saving visualization images to: {}'.format(save_path))

for img_id in range(start_id, end_id + 1):

    # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
    img_fname = src_files[img_id]
    img_fname_no_ext = os.path.splitext(img_fname)[0]

    src_img_fname = os.path.join(images_path, img_fname)
    src_img = imread(src_img_fname)
    if src_img is None:
        raise SystemError('Source image could not be read from: {}'.format(src_img_fname))

    stitched = src_img

    src_height, src_width, _ = src_img.shape

    if labels_path:
        labels_img_fname = os.path.join(labels_path, img_fname_no_ext + '.{}'.format(labels_ext))
        labels_img_orig = imread(labels_img_fname)
        if labels_img_orig is None:
            raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))
        if len(labels_img_orig.shape) == 3:
            labels_img_orig = np.squeeze(labels_img_orig[:, :, 0])

        if normalize_labels:
            labels_img = (labels_img_orig * label_diff).astype(np.uint8)
        else:
            labels_img = np.copy(labels_img_orig)

        if len(labels_img.shape) != 3:
            labels_img = np.stack((labels_img, labels_img, labels_img), axis=2)

        stitched = np.concatenate((stitched, labels_img), axis=1)

    for seg_dir in seg_dirs:
        seg_img_fname = os.path.join(seg_root_dir, seg_dir, img_fname_no_ext + '.{}'.format(seg_ext))
        seg_img = imread(seg_img_fname)
        if seg_img is None:
            raise SystemError('Segmentation image could not be read from: {}'.format(seg_img_fname))

        # if len(seg_img.shape) == 3:
        #     seg_img = np.squeeze(seg_img[:, :, 0])

        seg_height, seg_width, _ = seg_img.shape

        if seg_width == 2 * src_width or seg_width == 3 * src_width:
            _start_id = seg_width - src_width
            seg_img = seg_img[:, _start_id:, :]

        # print('seg_img.shape: ', seg_img.shape)
        # print('labels_img_orig.shape: ', labels_img_orig.shape)

        # seg_img = (seg_img * label_diff).astype(np.uint8)
        # if len(seg_img.shape) != 3:
        #     seg_img = np.stack((seg_img, seg_img, seg_img), axis=2)

        stitched = np.concatenate((stitched, seg_img), axis=1)
        if show_img:
            cv2.imshow('seg_img', seg_img)

    if save_stitched:
        seg_save_path = os.path.join(save_path, img_fname_no_ext + '.{}'.format(seg_ext))
        imsave(seg_save_path, stitched)
    if show_img:
        if resize_factor != 1:
            stitched = cv2.resize(stitched, (0,0), fx=resize_factor, fy=resize_factor)
        cv2.imshow('stitched', stitched)
        k = cv2.waitKey(1)
        if k == 27:
            break

    img_done = img_id - start_id + 1
    if img_done % 100 == 0:
        log_txt = '\rDone {:5d}/{:5d} frames'.format(img_done, n_frames)
        sys.stdout.write(log_txt)
        sys.stdout.flush()

sys.stdout.write('\n')
sys.stdout.flush()
