import os
import shutil
from datetime import datetime

import cv2
import numpy as np
import scipy
import pandas as pd
from PIL import Image

from tasks import task_utils


class Metrics:
    def __init__(self, class_id_to_name, class_id_to_col):
        self.pix_acc = 0
        self.fw_iu = 0
        self.miou = 0
        self.ca_miou = 0

        self.class_to_count = dict()
        self.acc = dict()
        self.iu = dict()
        self.dice = dict()
        self.class_id_to_name = class_id_to_name
        self.class_id_to_col = class_id_to_col

        # self.skip_acc = dict()
        # self.skip_iu = dict()

        self.n_classes = len(class_id_to_name)

        self.class_ids = []
        self.class_names = []
        for class_id, class_name in class_id_to_name.items():
            self.class_ids.append(class_id)
            self.class_names.append(class_name)

            # self.acc[class_name] = 0
            # self.iu[class_name] = 0
            # self.dice[class_name] = 0

            # self.skip_acc[class_name] = 0
            # self.skip_iu[class_name] = 0

        # sort_idx = np.argsort(self.class_ids)
        # self.class_ids = list(np.asarray(self.class_ids)[sort_idx])
        # self.class_names = list(np.asarray(self.class_names)[sort_idx])

        self.class_to_count['mean'] = 0

        if self.n_classes > 1:
            self.acc['mean'] = 0
            self.iu['mean'] = 0
            self.dice['mean'] = 0

        self.has_background = 0 in self.class_ids

    def update(self, pred_segm, gt_segm):
        check_size(pred_segm, gt_segm)

        n_classes_with_bkg = self.n_classes if self.has_background else self.n_classes + 1

        pred_segm[pred_segm >= n_classes_with_bkg] = 0

        pred_masks, gt_masks = extract_both_masks(pred_segm, gt_segm, self.class_ids)

        self.miou = compute_miou(pred_segm, gt_segm, n_classes_with_bkg)
        self.ca_miou = compute_class_agnostic_miou(pred_segm, gt_segm, pred_masks, gt_masks, self.has_background,
                                                   self.class_id_to_col)

        sum_intersection = 0
        sum_n_gt = 0
        sum_fw_IU = 0

        acc_list = []
        iu_list = []
        dice_list = []

        for i, c in enumerate(self.class_names):
            pred = pred_masks[i, :, :]
            gt = gt_masks[i, :, :]

            n_gt = np.sum(gt)
            n_pred = np.sum(pred)
            union_plus_intersection = n_gt + n_pred

            if union_plus_intersection == 0:
                """if neither GT nor pred of this class exists in this image, 
                metrics of this class are undefined for this image"""
                continue

            intersection = np.sum(np.logical_and(pred, gt))
            union = union_plus_intersection - intersection

            sum_intersection += intersection
            sum_n_gt += n_gt

            dice = intersection * 2.0 / union_plus_intersection
            acc = intersection / n_gt if n_gt > 0 else 0
            IU = intersection / union
            fw_IU = n_gt * IU

            sum_fw_IU += fw_IU
            self.acc[c] = acc
            self.iu[c] = IU
            self.dice[c] = dice

            acc_list.append(acc)
            iu_list.append(IU)
            dice_list.append(dice)

        if self.has_background:
            n_pix = get_pixel_area(pred_segm)
            assert sum_n_gt == n_pix, "sum_n_gt mismatch"

        self.fw_iu = sum_fw_IU / sum_n_gt
        self.pix_acc = sum_intersection / sum_n_gt

        if self.n_classes > 1:
            self.acc['mean'] = np.mean(acc_list)
            self.iu['mean'] = np.mean(iu_list)
            self.dice['mean'] = np.mean(dice_list)

        # print()

        # if self.n_frg_classes > 1:
        #     self.acc['frg'] = np.mean([acc for c, acc in zip(self.class_ids, acc_list) if c > 0])
        #     self.iu['frg'] = np.mean([iu for c, iu in zip(self.class_ids, iu_list) if c > 0])
        #     self.dice['frg'] = np.mean([dice for c, dice in zip(self.class_ids, dice_list) if c > 0])

    def to_dict(self):
        out_dict = dict(
            pix_acc=self.pix_acc,
            fw_iu=self.fw_iu,
            miou=self.miou,
            ca_miou=self.ca_miou,
        )
        for key, val in self.acc.items():
            out_dict[f'acc-{key}'] = val
        for key, val in self.iu.items():
            out_dict[f'iu-{key}'] = val
        for key, val in self.dice.items():
            out_dict[f'dice-{key}'] = val

        return out_dict

    def to_str(self, filter=None):
        if filter is None:
            filter = ['pix', 'acc', 'iu', 'dice']
        elif isinstance(filter, str):
            filter = [filter, ]
        str_ = ''
        if 'pix' in filter:
            str_ = (f"pix_acc: {self.pix_acc:.5f} fw_iu: {self.fw_iu:.5f} miou: {self.miou:.5f} ca_miou: "
                    f"{self.ca_miou:.5f}")
        if 'acc' in filter:
            for key, val in self.acc.items():
                str_ += f' acc-{key}: {val:.5f}'
        if 'iu' in filter:
            for key, val in self.iu.items():
                str_ += f' iu-{key}: {val:.5f}'
        if 'dice' in filter:
            for key, val in self.dice.items():
                str_ += f' dice-{key}: {val:.5f}'
        return str_

    def update_average(self, metrics, img_id):
        self.pix_acc += (metrics.pix_acc - self.pix_acc) / (img_id + 1)
        self.fw_iu += (metrics.fw_iu - self.fw_iu) / (img_id + 1)
        self.miou += (metrics.miou - self.miou) / (img_id + 1)
        self.ca_miou += (metrics.ca_miou - self.ca_miou) / (img_id + 1)

        for key in metrics.acc:
            if key in self.acc:
                self.class_to_count[key] += 1
                self.acc[key] += (metrics.acc[key] - self.acc[key]) / self.class_to_count[key]
                self.iu[key] += (metrics.iu[key] - self.iu[key]) / self.class_to_count[key]
                self.dice[key] += (metrics.dice[key] - self.dice[key]) / self.class_to_count[key]
            else:
                self.acc[key] = metrics.acc[key]
                self.iu[key] = metrics.iu[key]
                self.dice[key] = metrics.dice[key]
                self.class_to_count[key] = 1

                # print('_acc: {}'.format(_acc))
        # print('_IU: {}'.format(_IU))

        # mean_acc_ice = np.mean(list(_acc.values())[1:])
        # metrics.avg_mean_acc_ice += (mean_acc_ice - metrics.avg_mean_acc_ice) / (img_id + 1)

        # try:
        #     mean_acc_ice_1 = _acc[1]
        #     metrics.avg_mean_acc_ice_1 += (mean_acc_ice_1 - metrics.avg_mean_acc_ice_1) / (
        #             img_id - metrics.skip_mean_acc_ice_1 + 1)
        # except KeyError:
        #     print('\nskip_mean_acc_ice_1: {}'.format(img_id))
        #     metrics.skip_mean_acc_ice_1 += 1
        # try:
        #     mean_acc_ice_2 = _acc[2]
        #     metrics.avg_mean_acc_ice_2 += (mean_acc_ice_2 - metrics.avg_mean_acc_ice_2) / (
        #             img_id - metrics.skip_mean_acc_ice_2 + 1)
        # except KeyError:
        #     print('\nskip_mean_acc_ice_2: {}'.format(img_id))
        #     metrics.skip_mean_acc_ice_2 += 1
        #
        # mean_IU_ice = np.mean(list(_IU.values())[1:])
        # metrics.avg_mean_IU_ice += (mean_IU_ice - metrics.avg_mean_IU_ice) / (img_id + 1)
        # try:
        #     mean_IU_ice_1 = _IU[1]
        #     metrics.avg_mean_IU_ice_1 += (mean_IU_ice_1 - metrics.avg_mean_IU_ice_1) / (
        #             img_id - metrics.skip_mean_IU_ice_1 + 1)
        # except KeyError:
        #     print('\nskip_mean_IU_ice_1: {}'.format(img_id))
        #     metrics.skip_mean_IU_ice_1 += 1
        # try:
        #     mean_IU_ice_2 = _IU[2]
        #     metrics.avg_mean_IU_ice_2 += (mean_IU_ice_2 - metrics.avg_mean_IU_ice_2) / (
        #             img_id - metrics.skip_mean_IU_ice_2 + 1)
        # except KeyError:
        #     print('\nskip_mean_IU_ice_2: {}'.format(img_id))
        #     metrics.skip_mean_IU_ice_2 += 1


def metrics_to_csv(img_to_metrics, csv_path, csv_path_dup=None, write_info=False):
    img_to_metrics_df = pd.DataFrame.from_dict(img_to_metrics, orient='index')
    mean_df = img_to_metrics_df.mean(axis=0).to_frame('mean').transpose()
    median_df = img_to_metrics_df.median(axis=0).to_frame('median').transpose()
    img_to_metrics_df = pd.concat((mean_df, median_df, img_to_metrics_df))
    print(f'saving metrics to {csv_path}')

    csv_path_dir = os.path.dirname(csv_path)
    os.makedirs(csv_path_dir, exist_ok=True)

    with open(csv_path, 'w') as fid:
        img_to_metrics_df.to_csv(fid)

    if csv_path_dup is not None:
        csv_path_dup_dir = os.path.dirname(csv_path_dup)
        os.makedirs(csv_path_dup_dir, exist_ok=True)

        shutil.copy(csv_path, csv_path_dup)

        if write_info:
            csv_path_dup_name = os.path.splitext(os.path.basename(csv_path_dup))[0]
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
            info_path = os.path.join(csv_path_dup_dir, f"{csv_path_dup_name}-{time_stamp}.txt")
            with open(info_path, 'w') as fid:
                fid.write(csv_path)


def pixel_accuracy(eval_segm, gt_segm, cl):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = extract_classes(gt_segm)
    else:
        n_cl = len(cl)

    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if sum_t_i == 0:
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm, cl, return_acc):
    """
    (1/n_cl) sum_i(n_ii/t_i)
    """
    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = extract_classes(gt_segm)
    else:
        n_cl = len(cl)

    """class-specific binary masks"""
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if t_i != 0:
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    if return_acc:

        return dict(zip(cl, accuracy)), mean_accuracy_
    else:
        return mean_accuracy_


def mean_IU(eval_segm, gt_segm, cl, return_iu):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = union_classes(eval_segm, gt_segm)
    else:
        n_cl = len(cl)

    # _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        if t_i + n_ij == 0:
            IU[i] = 1
        elif t_i == 0 or n_ij == 0:
            IU[i] = 0
        else:
            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl
    if return_iu:
        return dict(zip(cl, IU)), mean_IU_
    else:
        return mean_IU_


def dice_score(eval_segm, gt_segm, cl, return_dice):
    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = union_classes(eval_segm, gt_segm)
    else:
        n_cl = len(cl)

    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    DS = list([0]) * n_cl

    for i, c in enumerate(cl):
        seg = eval_mask[i, :, :]
        gt = gt_mask[i, :, :]

        union = np.sum(seg) + np.sum(gt)
        if union == 0:
            dice = 1
        else:
            dice = np.sum(seg[gt == 1]) * 2.0 / union

        DS[i] = dice

    mean_DS_ = np.sum(DS) / n_cl_gt
    if return_dice:
        return dict(zip(cl, DS)), mean_DS_
    else:
        return mean_DS_


def frequency_weighted_IU(eval_segm, gt_segm, cl, return_freq):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = union_classes(eval_segm, gt_segm)
    else:
        n_cl = len(cl)

    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl
    t_i_list = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
        t_i_list[i] = t_i

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    if return_freq:
        return frequency_weighted_IU_, t_i_list
    else:
        return frequency_weighted_IU_


def compute_class_agnostic_miou(pred_segm, gt_segm, pred_masks, gt_masks,
                                has_background, class_id_to_col):
    gt_unique = np.unique(gt_segm, return_counts=False)
    pred_unique = np.unique(pred_segm, return_counts=False)

    gt_mask_ids = [k if has_background else k - 1 for k in gt_unique if k > 0]
    pred_mask_ids = [k if has_background else k - 1 for k in pred_unique if k > 0]

    n_gt_mask_ids, n_pred_mask_ids = len(gt_mask_ids), len(pred_mask_ids)
    pairwise_iou_cost = np.zeros((n_gt_mask_ids, n_pred_mask_ids), dtype=np.float32)

    for gt_id, gt_mask_id in enumerate(gt_mask_ids):
        gt_ = gt_masks[gt_mask_id, :, :]
        gt = gt_.astype(bool)
        for pred_id, pred_mask_id in enumerate(pred_mask_ids):
            pred_ = pred_masks[pred_mask_id, :, :]
            pred = pred_.astype(bool)
            intersection = np.sum(np.logical_and(pred, gt))
            if intersection == 0:
                iou = 0
            else:
                union = np.sum(np.logical_or(pred, gt))
                assert union > 0, "weird zero union between pred and gt"
                iou = intersection / union

            pairwise_iou_cost[gt_id, pred_id] = 1 - iou

    row_inds, col_inds = scipy.optimize.linear_sum_assignment(pairwise_iou_cost)

    row_col_inds = np.concatenate((row_inds, col_inds), axis=0)
    assert len(row_inds) == len(col_inds), "row_inds, col_inds len mismatch"
    gt_segm_matched = np.zeros_like(gt_segm)
    pred_segm_matched = np.zeros_like(pred_segm)

    for match_id, (row_ind, col_ind) in enumerate(zip(row_inds, col_inds, strict=True)):
        gt_matched_mask = gt_masks[gt_mask_ids[row_ind]].astype(bool)
        pred_matched_mask = pred_masks[pred_mask_ids[col_ind]].astype(bool)
        gt_segm_matched[gt_matched_mask] = match_id + 1
        pred_segm_matched[pred_matched_mask] = match_id + 1

    n_classes_with_bkg = n_gt_mask_ids + 1

    if n_gt_mask_ids > n_pred_mask_ids:
        n_classes_with_bkg = n_gt_mask_ids + 1
        assert len(row_inds) == len(col_inds) == n_pred_mask_ids, "n_pred_mask_ids mismatch"
        unmatched_gt_mask_ids = [k for i, k in enumerate(gt_mask_ids) if i not in row_inds]
        assert len(unmatched_gt_mask_ids) == n_gt_mask_ids - n_pred_mask_ids, \
            "unmatched_gt_mask_ids len mismatch"
        for _id, unmatched_gt_mask_id in enumerate(unmatched_gt_mask_ids):
            gt_mask = gt_masks[unmatched_gt_mask_id].astype(bool)
            gt_segm_matched[gt_mask] = _id + n_pred_mask_ids + 1

    elif n_gt_mask_ids < n_pred_mask_ids:
        n_classes_with_bkg = n_pred_mask_ids + 1

        assert len(row_inds) == len(col_inds) == n_gt_mask_ids, "n_gt_mask_ids mismatch"
        unmatched_pred_mask_ids = [k for i, k in enumerate(pred_mask_ids) if i not in col_inds]
        assert len(unmatched_pred_mask_ids) == n_pred_mask_ids - n_gt_mask_ids, \
            "unmatched_pred_mask_ids len mismatch"
        for _id, unmatched_pred_mask_id in enumerate(unmatched_pred_mask_ids):
            pred_mask = pred_masks[unmatched_pred_mask_id].astype(bool)
            pred_segm_matched[pred_mask] = _id + n_gt_mask_ids + 1

    ca_miou = compute_miou(pred_segm_matched, gt_segm_matched, n_classes_with_bkg)

    # pred_segm_vis = task_utils.mask_id_to_vis_bgr(pred_segm, class_id_to_col)
    # gt_segm_vis = task_utils.mask_id_to_vis_bgr(gt_segm, class_id_to_col)
    # cmb_segm_vis = np.concatenate((gt_segm_vis, pred_segm_vis), axis=1)
    #
    # pred_segm_matched_vis = task_utils.mask_id_to_vis_bgr(pred_segm_matched, class_id_to_col)
    # gt_segm_matched_vis = task_utils.mask_id_to_vis_bgr(gt_segm_matched, class_id_to_col)
    # cmb_matched_segm_vis = np.concatenate((gt_segm_matched_vis, pred_segm_matched_vis), axis=1)
    #
    # cv2.imshow('cmb_segm_vis', cmb_segm_vis)
    # cv2.imshow('cmb_matched_segm_vis', cmb_matched_segm_vis)
    # cv2.waitKey(0)

    return ca_miou


def compute_miou(pred_segm, gt_segm, n_classes_with_bkg):
    """
    algorithm from: https://medium.com/@cyborg.team.nitr/miou-calculation-4875f918f4cb

    :param gt_segm:
    :param pred_segm:
    :param n_classes_with_bkg:
    :return:
    """
    gt_1d = gt_segm.astype(np.int64).flatten()
    pred_1d = pred_segm.astype(np.int64).flatten()

    gt_unique, gt_counts = np.unique(gt_segm, return_counts=True)
    pred_unique, pred_counts = np.unique(pred_segm, return_counts=True)

    gt_count = np.zeros((n_classes_with_bkg, 1), dtype=np.int64)
    for category_id, category_count in zip(gt_unique, gt_counts, strict=True):
        gt_count[category_id] = category_count

    pred_count = np.zeros((n_classes_with_bkg, 1), dtype=np.int64)
    for category_id, category_count in zip(pred_unique, pred_counts, strict=True):
        if category_id >= n_classes_with_bkg:
            """ignore invalid class labels in pred"""
            # continue
            """treat invalid class labels as background"""
            category_id = 0

        pred_count[category_id] = category_count

    category_1d = (n_classes_with_bkg * gt_1d) + pred_1d

    category_1d_unique, category_1d_counts = np.unique(category_1d, return_counts=True)
    cm_1d = np.zeros((n_classes_with_bkg * n_classes_with_bkg, 1), dtype=np.int64)

    for category_id, category_count in zip(category_1d_unique, category_1d_counts, strict=True):
        cm_1d[category_id] = category_count

    cm_2d = np.reshape(cm_1d, (n_classes_with_bkg, n_classes_with_bkg))
    I = np.diagonal(cm_2d)

    U = gt_count.squeeze() + pred_count.squeeze() - I

    class_ious = np.divide(I, U)
    miou = np.nanmean(class_ious)

    return miou


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, class_ids):
    eval_mask = extract_masks(eval_segm, class_ids)
    gt_mask = extract_masks(gt_segm, class_ids)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, class_ids):
    h, w = segm_size(segm)
    masks = np.zeros((len(class_ids), h, w))

    for i, c in enumerate(class_ids):
        if isinstance(c, (tuple, list)):
            masks[i, :, :] = np.logical_or.reduce([segm == c_ for c_ in c])

        elif isinstance(c, int):
            masks[i, :, :] = segm == c
        else:
            raise AssertionError(f'invalid class id: {c}')

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
