import argparse, os, sys
import copy

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import cv2
import time
from PIL import Image

from tqdm import tqdm
from pprint import pprint, pformat
from tabulate import tabulate

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import paramparse

# plt.style.use('presentation')

import densenet.evaluation.eval_segm as eval
from densenet.utils import read_data, linux_path, print_and_write, resize_ar, put_text_with_background, col_bgr
from dictances import bhattacharyya, euclidean, mae, mse


class Params:
    """
    :ivar codec:
    :type codec: str

    :ivar enable_plotting: enable_plotting
    :type enable_plotting: int

    :ivar end_id:
    :type end_id: int

    :ivar fps:
    :type fps: float

    :ivar ice_type: 0: combined, 1: anchor, 2: frazil
    :type ice_type: int

    :ivar images_ext:
    :type images_ext: str

    :ivar images_path:
    :type images_path: str

    :ivar labels_col:
    :type labels_col: str

    :ivar labels_ext:
    :type labels_ext: str

    :ivar labels_path:
    :type labels_path: str

    :ivar load_ice_conc_diff:
    :type load_ice_conc_diff: int

    :ivar log_dir:
    :type log_dir: str

    :ivar n_classes:
    :type n_classes: int

    :ivar normalize_labels:
    :type normalize_labels: int

    :ivar out_ext:
    :type out_ext: str

    :ivar out_path:
    :type out_path: str

    :ivar out_size:
    :type out_size: str

    :ivar plot_changed_seg_count:
    :type plot_changed_seg_count: int

    :ivar save_path:
    :type save_path: str

    :ivar save_stitched:
    :type save_stitched: int

    :ivar seg_cols:
    :type seg_cols: str_to_list

    :ivar seg_ext:
    :type seg_ext: str

    :ivar seg_labels:
    :type seg_labels: str_to_list

    :ivar seg_paths:
    :type seg_paths: str_to_list

    :ivar seg_root_dir:
    :type seg_root_dir: str

    :ivar selective_mode:
    :type selective_mode: int

    :ivar show_img:
    :type show_img: int

    :ivar start_id:
    :type start_id: int

    :ivar stitch:
    :type stitch: int

    :ivar stitch_seg:
    :type stitch_seg: int

    """

    def __init__(self):
        self.cfg = ()
        self.codec = 'H264'
        self.enable_plotting = 1
        self.end_id = -1
        self.fps = 30
        self.ice_type = 0

        self.resize = 0

        self.db_path = ''

        self.images_ext = 'png'
        self.images_path = ''

        self.labels_col = 'green'
        self.labels_ext = 'png'
        self.labels_path = ''

        self.load_ice_conc_diff = 0
        self.log_dir = ''
        self.n_classes = 3
        self.normalize_labels = 0

        self.out_ext = 'jpg'
        self.out_path = ''
        self.out_size = '1920x1080'

        self.plot_changed_seg_count = 0
        self.save_path = ''
        self.save_stitched = 0
        self.seg_cols = ['blue', 'forest_green', 'magenta', 'cyan', 'red']
        self.seg_ext = 'png'
        self.seg_labels = []
        self.seg_paths = []
        self.seg_root_dir = ''
        self.selective_mode = 0
        self.show_img = 0
        self.start_id = 0
        self.stitch = 0
        self.stitch_seg = 1


def get_plot_image(data_x, data_y, cols, title, line_labels, x_label, y_label, ylim=None, legend=1):
    cols = [(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0) for col in cols]

    fig = Figure(
        # figsize=(6.4, 3.6), dpi=300,
        figsize=(4.8, 2.7), dpi=400,
        # edgecolor='k',
        # facecolor ='k'
    )
    # fig.tight_layout()
    # fig.set_tight_layout(True)
    fig.subplots_adjust(
        bottom=0.17,
        right=0.95,
    )
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    n_data = len(data_y)
    for i in range(n_data):
        datum_y = data_y[i]
        line_label = line_labels[i]
        col = cols[i]
        args = {
            'color': col
        }
        if legend:
            args['label'] = line_label
        ax.plot(data_x, datum_y, **args)
    plt.rcParams['axes.titlesize'] = 10
    # fontdict = {'fontsize': plt.rcParams['axes.titlesize'],
    #             'fontweight': plt.rcParams['axes.titleweight'],
    # 'verticalalignment': 'baseline',
    # 'horizontalalignment': plt.loc
    # }
    ax.set_title(title,
                 # fontdict=fontdict
                 )
    if legend:
        ax.legend(fancybox=True, framealpha=0.1)
    ax.grid(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if ylim is not None:
        ax.set_ylim(*ylim)

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plot_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)

    return plot_img


def str_to_list(_str, _type=str, _sep=','):
    if _sep not in _str:
        _str += _sep
    return [k for k in list(map(_type, _str.split(_sep))) if k]


def main():
    params: Params = paramparse.process(Params)
    for ice_type in ice_types:
        params_ = copy.deepcopy(params)
        params_.ice_type = ice_type
        run(params_)


def run(params):
    db_path = params.db_path
    images_path = params.images_path
    images_ext = params.images_ext
    labels_path = params.labels_path
    labels_ext = params.labels_ext
    labels_col = params.labels_col

    seg_paths = params.seg_paths
    seg_root_dir = params.seg_root_dir
    seg_ext = params.seg_ext

    out_path = params.out_path
    out_ext = params.out_ext
    out_size = params.out_size
    fps = params.fps
    codec = params.codec

    # save_path = args.save_path

    n_classes = params.n_classes

    end_id = params.end_id
    start_id = params.start_id

    show_img = params.show_img
    stitch = params.stitch
    stitch_seg = params.stitch_seg
    save_stitched = params.save_stitched

    normalize_labels = params.normalize_labels
    selective_mode = params.selective_mode

    seg_labels = params.seg_labels
    seg_cols = params.seg_cols

    ice_type = params.ice_type

    plot_changed_seg_count = params.plot_changed_seg_count

    load_ice_conc_diff = params.load_ice_conc_diff
    enable_plotting = params.enable_plotting

    loc = (5, 120)
    size = 8
    thickness = 6
    fgr_col = (255, 255, 255)
    bgr_col = (0, 0, 0)
    font_id = 0

    video_exts = ['mp4', 'mkv', 'avi', 'mpg', 'mpeg', 'mjpg']

    labels_col_rgb = col_bgr[labels_col]
    seg_cols_rgb = [col_bgr[seg_col] for seg_col in seg_cols]

    ice_type_str = ice_types[ice_type]

    print('ice_type_str: {}'.format(ice_type_str))

    if db_path:
        if not images_path:
            images_path = 'images'
        images_path = linux_path(db_path, images_path)
        if labels_path:
            labels_path = linux_path(db_path, labels_path)

    src_files, src_labels_list, total_frames = read_data(
        images_path, images_ext, labels_path, labels_ext)
    if end_id < start_id:
        end_id = total_frames - 1

    if seg_paths:
        n_seg_paths = len(seg_paths)
        n_seg_labels = len(seg_labels)

        if n_seg_paths != n_seg_labels:
            raise IOError('Mismatch between n_seg_labels: {} and n_seg_paths: {}'.format(
                n_seg_labels, n_seg_paths
            ))
        if seg_root_dir:
            seg_paths = [linux_path(seg_root_dir, name) for name in seg_paths]

    if not out_path:
        if seg_paths:
            out_path = seg_paths[0] + '_conc'
        elif labels_path:
            out_path = labels_path + '_conc'

        os.makedirs(out_path, exist_ok=True)

    # print('Saving results data to {}'.format(out_path))

    # if not save_path:
    #     save_path = linux_path(os.path.dirname(images_path), 'ice_concentration')
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)

    # if stitch and save_stitched:
    #     print('Saving ice_concentration plots to: {}'.format(save_path))

    # log_fname = linux_path(out_path, 'vis_log_{:s}.txt'.format(getDateTime()))
    # print('Saving log to: {}'.format(log_fname))

    if selective_mode:
        label_diff = int(255.0 / n_classes)
    else:
        label_diff = int(255.0 / (n_classes - 1))

    print('label_diff: {}'.format(label_diff))

    n_frames = end_id - start_id + 1

    print_diff = int(n_frames * 0.01)

    labels_img = None

    n_cols = len(seg_cols_rgb)

    plot_y_label = '{} concentration (%)'.format(ice_type_str)
    plot_x_label = 'distance in pixels from left edge'

    dists = {}

    for _label in seg_labels:
        dists[_label] = {
            # 'bhattacharyya': [],
            'euclidean': [],
            'mae': [],
            'mse': [],
            # 'frobenius': [],
        }

    plot_title = '{} concentration'.format(ice_type_str)

    out_size = tuple([int(x) for x in out_size.split('x')])
    write_to_video = out_ext in video_exts
    out_width, out_height = out_size

    out_seq_name = os.path.basename(out_path)

    if enable_plotting:
        if write_to_video:
            stitched_seq_path = linux_path(out_path, '{}.{}'.format(out_seq_name, out_ext))
            print('Writing {}x{} output video to: {}'.format(out_width, out_height, stitched_seq_path))
            save_dir = os.path.dirname(stitched_seq_path)

            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_out = cv2.VideoWriter(stitched_seq_path, fourcc, fps, out_size)
        else:
            stitched_seq_path = linux_path(out_path, out_seq_name)
            print('Writing {}x{} output images of type {} to: {}'.format(
                out_width, out_height, out_ext, stitched_seq_path))
            save_dir = stitched_seq_path

        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    prev_seg_img = {}
    prev_conc_data_y = {}

    changed_seg_count = {}
    ice_concentration_diff = {}

    if load_ice_conc_diff:
        for seg_id in seg_labels:
            ice_concentration_diff[seg_id] = np.loadtxt(
                linux_path(out_path, '{}_ice_concentration_diff.txt'.format(seg_id)),
                dtype=np.float64)
    _pause = 0

    mae_data_y = []
    for seg_id, _ in enumerate(seg_paths):
        mae_data_y.append([])

    df_rows = []
    for img_id in tqdm(range(start_id, end_id + 1), position=0, leave=True):
        # img_fname = '{:s}_{:d}.{:s}'.format(fname_templ, img_id + 1, img_ext)
        img_path = src_files[img_id]
        img_fname = os.path.basename(img_path)
        img_fname_no_ext = os.path.splitext(img_fname)[0]
        df_rows.append(img_fname_no_ext)

        src_img_fname = linux_path(images_path, img_fname)
        src_img_pil = Image.open(src_img_fname)
        if params.resize:
            src_img_pil = src_img_pil.resize((params.resize, params.resize))
        src_img = np.array(src_img_pil)

        try:
            src_height, src_width = src_img.shape[:2]
        except ValueError as e:
            print('src_img_fname: {}'.format(src_img_fname))
            print('src_img: {}'.format(src_img))
            print('src_img.shape: {}'.format(src_img.shape))
            print('error: {}'.format(e))
            sys.exit(1)

        conc_data_x = np.asarray(range(src_width), dtype=np.float64)
        plot_data_x = conc_data_x

        plot_data_y = []
        plot_cols = []

        plot_labels = []

        stitched_img = src_img

        if labels_path:
            labels_img_path = linux_path(labels_path, img_fname_no_ext + '.{}'.format(labels_ext))

            gt_labels_pil = Image.open(labels_img_path)
            if params.resize:
                gt_labels_pil = gt_labels_pil.resize((params.resize, params.resize))
            labels_img_orig = np.array(gt_labels_pil)

            labels_height, labels_width = labels_img_orig.shape[:2]

            assert labels_height == src_height and labels_width == src_width, (
                'Mismatch between dimensions of source: {} and label: {}'.format(
                    (src_height, src_width), (labels_height, labels_width)))

            if len(labels_img_orig.shape) == 3:
                labels_img_orig = np.squeeze(labels_img_orig[:, :, 0])

            if show_img:
                cv2.imshow('labels_img_orig', labels_img_orig)

            if normalize_labels:
                labels_img = (labels_img_orig.astype(np.float64) / label_diff).astype(np.uint8)
            else:
                labels_img = np.copy(labels_img_orig)

            if len(labels_img.shape) == 3:
                labels_img = labels_img[:, :, 0].squeeze()

            conc_data_y = np.zeros((labels_width,), dtype=np.float64)

            for i in range(labels_width):
                curr_pix = np.squeeze(labels_img[:, i])
                if ice_type == 0:
                    ice_pix = curr_pix[curr_pix != 0]
                else:
                    ice_pix = curr_pix[curr_pix == ice_type]

                conc_data_y[i] = (len(ice_pix) / float(src_height)) * 100.0

            conc_data = np.zeros((labels_width, 2), dtype=np.float64)
            conc_data[:, 0] = conc_data_x
            conc_data[:, 1] = conc_data_y

            plot_data_y.append(conc_data_y)
            plot_cols.append(labels_col_rgb)

            gt_dict = {conc_data_x[i]: conc_data_y[i] for i in range(labels_width)}

            if not normalize_labels:
                labels_img_orig = (labels_img_orig.astype(np.float64) * label_diff).astype(np.uint8)

            if len(labels_img_orig.shape) == 2:
                labels_img_orig = np.stack((labels_img_orig, labels_img_orig, labels_img_orig), axis=2)

            stitched_img = np.concatenate((stitched_img, labels_img_orig), axis=1)

            plot_labels.append('GT')

            # gt_cl, _ = eval.extract_classes(labels_img_orig)
            # print('gt_cl: {}'.format(gt_cl))

        mean_seg_counts = {}
        seg_count_data_y = []
        curr_mae_data_y = []

        mean_conc_diff = {}
        conc_diff_data_y = []
        seg_img_disp_list = []

        for seg_id, seg_path in enumerate(seg_paths):
            seg_img_path = linux_path(seg_path, img_fname_no_ext + '.{}'.format(seg_ext))
            gt_labels_pil = Image.open(seg_img_path)
            seg_img_orig = np.array(gt_labels_pil)

            # seg_img_orig = imread(seg_img_path)

            seg_col = seg_cols_rgb[seg_id % n_cols]

            _label = seg_labels[seg_id]

            if seg_img_orig is None:
                raise SystemError('Seg image could not be read from: {}'.format(seg_img_path))
            seg_height, seg_width = seg_img_orig.shape[:2]

            if seg_height != src_height or seg_width != src_width:
                raise AssertionError('Mismatch between dimensions of source: {} and seg: {}'.format(
                    (src_height, src_width), (seg_height, seg_width)
                ))

            if len(seg_img_orig.shape) == 3:
                seg_img_orig = np.squeeze(seg_img_orig[:, :, 0])

            if seg_img_orig.max() > n_classes - 1:
                seg_img = (seg_img_orig.astype(np.float64) / label_diff).astype(np.uint8)
                seg_img_disp = seg_img_orig
            else:
                seg_img = seg_img_orig
                seg_img_disp = (seg_img_orig.astype(np.float64) * label_diff).astype(np.uint8)

            if len(seg_img_disp.shape) == 2:
                seg_img_disp = np.stack((seg_img_disp, seg_img_disp, seg_img_disp), axis=2)

            ann_fmt = (font_id, loc[0], loc[1], size, thickness) + fgr_col + bgr_col
            put_text_with_background(seg_img_disp, seg_labels[seg_id], fmt=ann_fmt)

            seg_img_disp_list.append(seg_img_disp)
            # eval_cl, _ = eval.extract_classes(seg_img)
            # print('eval_cl: {}'.format(eval_cl))

            if show_img:
                cv2.imshow('seg_img_orig', seg_img_orig)

            if len(seg_img.shape) == 3:
                seg_img = seg_img[:, :, 0].squeeze()

            conc_data_y = np.zeros((seg_width,), dtype=np.float64)
            for i in range(seg_width):
                curr_pix = np.squeeze(seg_img[:, i])
                if ice_type == 0:
                    ice_pix = curr_pix[curr_pix != 0]
                else:
                    ice_pix = curr_pix[curr_pix == ice_type]
                conc_data_y[i] = (len(ice_pix) / float(src_height)) * 100.0

            plot_cols.append(seg_col)
            plot_data_y.append(conc_data_y)

            if labels_path:
                seg_dict = {conc_data_x[i]: conc_data_y[i] for i in range(seg_width)}
                # dists['bhattacharyya'].append(bhattacharyya(gt_dict, seg_dict))
                dists[_label]['euclidean'].append(euclidean(gt_dict, seg_dict))
                dists[_label]['mse'].append(mse(gt_dict, seg_dict))
                dists[_label]['mae'].append(mae(gt_dict, seg_dict))
                # dists['frobenius'].append(np.linalg.norm(conc_data_y - plot_data_y[0]))
                curr_mae_data_y.append(dists[_label]['mae'][-1])
            else:
                if img_id > 0:
                    if plot_changed_seg_count:
                        flow = cv2.calcOpticalFlowFarneback(prev_seg_img[_label], seg_img, None, 0.5, 3, 15, 3, 5, 1.2,
                                                            0)

                        print('flow: {}'.format(flow.shape))

                        # # Obtain the flow magnitude and direction angle
                        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        # hsvImg = np.zeros((2160, 3840, 3), dtype=np.uint8)
                        # hsvImg[..., 1] = 255
                        # # Update the color image
                        # hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
                        # hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        # rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
                        # rgbImg = resizeAR(rgbImg, width=out_width, height=out_height)
                        # # Display the resulting frame
                        # cv2.imshow('dense optical flow', rgbImg)
                        # k = cv2.waitKey(0)

                        curr_x, curr_y = (prev_x + flow[..., 0]).astype(np.int32), (prev_y + flow[..., 1]).astype(
                            np.int32)

                        seg_img_flow = seg_img[curr_y, curr_x]

                        changed_seg_count[_label].append(np.count_nonzero(np.not_equal(seg_img, prev_seg_img[_label])))
                        seg_count_data_y.append(changed_seg_count[_label])
                        mean_seg_counts[_label] = np.mean(changed_seg_count[_label])
                    else:
                        ice_concentration_diff[_label].append(np.mean(np.abs(conc_data_y - prev_conc_data_y[_label])))
                        conc_diff_data_y.append(ice_concentration_diff[_label])
                        mean_conc_diff[_label] = np.mean(ice_concentration_diff[_label])
                else:
                    if plot_changed_seg_count:
                        prev_x, prev_y = np.meshgrid(range(seg_width), range(seg_height), sparse=False, indexing='xy')
                        changed_seg_count[_label] = []
                    else:
                        ice_concentration_diff[_label] = []

            prev_seg_img[_label] = seg_img
            prev_conc_data_y[_label] = conc_data_y

        # conc_data = np.concatenate([conc_data_x, conc_data_y], axis=1)
        if labels_path:
            for i, k in enumerate(curr_mae_data_y):
                mae_data_y[i].append(k)

            n_test_images = img_id + 1
            mae_data_X = np.asarray(range(1, n_test_images + 1), dtype=np.float64)
            # print('mae_data_X:\n {}'.format(pformat(mae_data_X)))
            # print('mae_data_y:\n {}'.format(pformat(np.array(mae_data_y).transpose())))

            if img_id == end_id:
                mae_data_y_out = copy.deepcopy(mae_data_y)
                for mae_data_y_, label_ in zip(mae_data_y_out, seg_labels, strict=True):
                    mean_ = np.mean(mae_data_y_)
                    median_ = np.median(mae_data_y_)
                    mae_data_y_ += [mean_, median_]

                mae_data_y_arr = np.array(mae_data_y_out).transpose()
                # print('mae_data_y:\n {}'.format(tabulate(mae_data_y_arr,
                #                                          headers=seg_labels, tablefmt='plain')))
                df_rows += ['mean', 'median']

                mae_data_y_df = pd.DataFrame(data=mae_data_y_arr, columns=seg_labels, index=df_rows)
                # mae_data_y_df.to_clipboard(excel=True)
                mae_data_y_xlsx = linux_path(out_path, f'mae_data_y-{ice_type_str}.xlsx')
                print(f'\nsaving mae_data_y_df to {mae_data_y_xlsx}')
                mae_data_y_df.to_excel(mae_data_y_xlsx)

            mae_img = get_plot_image(mae_data_X, mae_data_y, plot_cols, 'MAE', seg_labels,
                                     'frame', 'MAE')
            if show_img:
                cv2.imshow('mae_img', mae_img)

            conc_diff_img = resize_ar(mae_img, seg_width, src_height, bkg_col=255)
        else:
            if img_id > 0:
                n_test_images = img_id
                seg_count_data_X = np.asarray(range(1, n_test_images + 1), dtype=np.float64)

                if plot_changed_seg_count:
                    seg_count_img = get_plot_image(seg_count_data_X, seg_count_data_y, plot_cols, 'Count', seg_labels,
                                                   'frame', 'Changed Label Count')
                    if show_img:
                        cv2.imshow('seg_count_img', seg_count_img)
                else:

                    # print('seg_count_data_X:\n {}'.format(pformat(seg_count_data_X)))
                    # print('conc_diff_data_y:\n {}'.format(pformat(conc_diff_data_y)))

                    conc_diff_img = get_plot_image(seg_count_data_X, conc_diff_data_y, plot_cols,
                                                   'Mean concentration difference between consecutive frames'.format(
                                                       ice_type_str),
                                                   seg_labels, 'frame', 'Concentration Difference (%)')
                    # cv2.imshow('conc_diff_img', conc_diff_img)
                    conc_diff_img = resize_ar(conc_diff_img, seg_width, src_height, bkg_col=255)
            else:
                conc_diff_img = np.zeros((src_height, seg_width, 3), dtype=np.uint8)

        plot_labels += seg_labels
        if enable_plotting:
            plot_img = get_plot_image(plot_data_x, plot_data_y, plot_cols, plot_title, plot_labels,
                                      plot_x_label, plot_y_label,
                                      legend=0
                                      # ylim=(0, 100)
                                      )

            plot_img = resize_ar(plot_img, seg_width, src_height, bkg_col=255)

            # plt.plot(conc_data_x, conc_data_y)
            # plt.show()

            # conc_data_fname = linux_path(out_path, img_fname_no_ext + '.txt')
            # np.savetxt(conc_data_fname, conc_data, fmt='%.6f')
            ann_fmt = (font_id, loc[0], loc[1], size, thickness) + labels_col_rgb + bgr_col

            put_text_with_background(src_img, 'frame {}'.format(img_id + 1), fmt=ann_fmt)

            if n_seg_paths == 1:
                # print('seg_img_disp: {}'.format(seg_img_disp.shape))
                # print('plot_img: {}'.format(plot_img.shape))
                stitched_seg_img = np.concatenate((seg_img_disp, plot_img), axis=1)

                # print('stitched_seg_img: {}'.format(stitched_seg_img.shape))
                # print('stitched_img: {}'.format(stitched_img.shape))
                stitched_img = np.concatenate((stitched_img, stitched_seg_img), axis=0 if labels_path else 1)
            elif n_seg_paths == 2:
                stitched_img = np.concatenate((
                    np.concatenate((src_img, conc_diff_img), axis=1),
                    np.concatenate(seg_img_disp_list, axis=1),
                ), axis=0)
            elif n_seg_paths == 3:
                stitched_img = np.concatenate((
                    np.concatenate((src_img, plot_img, conc_diff_img), axis=1),
                    np.concatenate(seg_img_disp_list, axis=1),
                ), axis=0)

            stitched_img = resize_ar(stitched_img, width=out_width, height=out_height)

            # print('dists: {}'.format(dists))

            if write_to_video:
                video_out.write(stitched_img)
            else:
                stacked_img_path = linux_path(stitched_seq_path, '{}.{}'.format(img_fname_no_ext, out_ext))
                cv2.imwrite(stacked_img_path, stitched_img)

            if show_img:
                cv2.imshow('stitched_img', stitched_img)
                k = cv2.waitKey(1 - _pause)
                if k == 27:
                    break
                elif k == 32:
                    _pause = 1 - _pause

    if enable_plotting and write_to_video:
        video_out.release()

    if labels_path:
        median_dists = {}
        mean_dists = {}
        mae_data_y = []
        for _label in seg_labels:
            _dists = dists[_label]
            mae_data_y.append(_dists['mae'])
            mean_dists[_label] = {k: np.mean(_dists[k]) for k in _dists}
            median_dists[_label] = {k: np.median(_dists[k]) for k in _dists}

        print('mean_dists:\n{}'.format(pformat(mean_dists)))
        print('median_dists:\n{}'.format(pformat(median_dists)))

        if show_img:
            n_test_images = len(mae_data_y[0])
            mae_data_x = np.asarray(range(1, n_test_images + 1), dtype=np.float64)
            mae_img = get_plot_image(mae_data_x, mae_data_y, plot_cols, 'MAE', seg_labels,
                                     'test image', 'Mean Absolute Error')
            # plt.show()
            cv2.imshow('MAE', mae_img)
            k = cv2.waitKey(0)
    else:
        mean_seg_counts = {}
        median_seg_counts = {}
        seg_count_data_y = []

        mean_conc_diff = {}
        median_conc_diff = {}
        conc_diff_data_y = []

        for seg_id in ice_concentration_diff:
            if plot_changed_seg_count:
                seg_count_data_y.append(changed_seg_count[seg_id])
                mean_seg_counts[seg_id] = np.mean(changed_seg_count[seg_id])
                median_seg_counts[seg_id] = np.median(changed_seg_count[seg_id])
            else:
                _ice_concentration_diff = ice_concentration_diff[seg_id]
                n_test_images = len(_ice_concentration_diff)

                conc_diff_data_y.append(_ice_concentration_diff)
                mean_conc_diff[seg_id] = np.mean(_ice_concentration_diff)
                median_conc_diff[seg_id] = np.median(_ice_concentration_diff)

                np.savetxt(linux_path(out_path, '{}_ice_concentration_diff.txt'.format(seg_id)),
                           _ice_concentration_diff, fmt='%8.4f', delimiter='\t')

        if plot_changed_seg_count:
            print('mean_seg_counts:\n{}'.format(pformat(mean_seg_counts)))
            print('median_seg_counts:\n{}'.format(pformat(median_seg_counts)))
        else:
            print('mean_conc_diff:')
            for seg_id in mean_conc_diff:
                print('{}\t{}'.format(seg_id, mean_conc_diff[seg_id]))
            print('median_conc_diff:')
            for seg_id in mean_conc_diff:
                print('{}\t{}'.format(seg_id, median_conc_diff[seg_id]))

            # seg_count_data_X = np.asarray(range(1, n_test_images + 1), dtype=np.float64)
            #
            # if plot_changed_seg_count:
            #     seg_count_img = getPlotImage(seg_count_data_X, seg_count_data_y, plot_cols, 'Count', seg_labels,
            #                                  'test image', 'Changed Label Count')
            #     cv2.imshow('seg_count_img', seg_count_img)
            #
            # conc_diff_img = getPlotImage(seg_count_data_X, conc_diff_data_y, plot_cols, 'Difference', seg_labels,
            #                              'test image', 'Concentration Difference')
            # cv2.imshow('conc_diff_img', conc_diff_img)

            # k = cv2.waitKey(0)


if __name__ == '__main__':
    ice_types = {
        0: 'Ice',
        1: 'Anchor Ice',
        2: 'Frazil Ice',
    }
    main()
