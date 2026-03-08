import os
import sys

import shlex
import subprocess
from datetime import datetime

import copy

p2s_path = os.path.join(os.path.expanduser("~"), "pix2seq")
sys.path.append(p2s_path)

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
sys.path.append(dproc_path)

import paramparse
from paramparse import MultiPath

from densenet.utils import linux_path
from tasks import task_utils

import sub_patch as spd


class BatchParams:
    def __init__(self):
        self.dataset = ''

        self.split = MultiPath()
        self.split_suffix = ''
        self.seq_id = -1
        self.seq_start_id = 0
        self.seq_end_id = -1
        self.n_rot = 0

        self.image_dir = ''
        self.labels_dir = ''
        self.instances_dir = 'instances'

        self.check_labels = 0
        self.n_proc = 0

        self.log_to_file = 0
        self.log_dir = 'log'


def run_cmd(no_rot_cmd, parallel, log_to_file, log_dir, processes):
    if parallel:
        args = shlex.split(no_rot_cmd)
        p = subprocess.Popen(args)
        if log_to_file:
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
            tee_log_id = 'sub_patch_batch_{}'.format(time_stamp)

            out_fname = tee_log_id + '.ansi'
            zip_fname = out_fname.replace('.ansi', '.zip')

            out_path = linux_path(log_dir, out_fname)

            f = open(out_path, 'w')
            p = subprocess.Popen(args, stdout=f, stderr=f)
        else:
            f = out_fname = zip_fname = None
            p = subprocess.Popen(args)

        processes.append((p, f, out_fname, zip_fname))
    else:
        subprocess.check_call(no_rot_cmd, shell=True)


def save_json(json_dict, json_path, json_gz=True):
    n_json_imgs = len(json_dict['images'])
    n_json_objs = len(json_dict['annotations'])
    if json_gz:
        json_path += '.gz'

    print(f'saving output json with {n_json_imgs} images and {n_json_objs} objects to: {json_path}')
    json_kwargs = dict(
        indent=4
    )
    if json_gz:
        import compress_json
        compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
    else:
        import json

        with open(json_path, 'w') as f:
            output_json_data = json.dumps(json_dict, **json_kwargs)
            f.write(output_json_data)


def set_output_paths(batch_params: BatchParams, spd_params: spd.Params, out_seq_name, db_root_dir):
    spd_params.out_seq_name = out_seq_name
    dataset = batch_params.dataset.lower()
    if dataset in ['617', '617_val', 'ipsc', 'coco', 'cityscapes']:
        spd_params.out_root_dir = f'{db_root_dir}-{spd_params.out_suffix}'
        spd_params.out_img_dir = linux_path(spd_params.out_root_dir, out_seq_name)
        spd_params.out_labels_dir = linux_path(spd_params.out_img_dir, batch_params.labels_dir)
        spd_params.out_instance_dir = linux_path(spd_params.out_img_dir, batch_params.instances_dir)
    elif dataset == 'ctc':
        spd_params.out_root_dir = f'{db_root_dir}-{spd_params.out_suffix}'
        spd_params.out_img_dir = linux_path(spd_params.out_root_dir, batch_params.image_dir, out_seq_name)
        spd_params.out_labels_dir = linux_path(spd_params.out_root_dir, batch_params.labels_dir, out_seq_name)
        spd_params.out_instance_dir = linux_path(spd_params.out_root_dir, batch_params.instances_dir, out_seq_name)
    # elif dataset == 'coco':
    #     spd_params.out_root_dir = linux_path(db_root_dir, spd_params.out_suffix)
    else:
        raise AssertionError(f'invalid dataset: {batch_params.dataset}')


def get_spd_params(spd_params, batch_params, all_params):
    """

    :param spd.Params spd_params:
    :param BatchParams batch_params:
    :return:
    """

    # py_exe = spd_params.py_exe
    # if not py_exe:
    #     py_exe = sys.executable

    spd_params = copy.deepcopy(spd_params)

    db_root_dir = spd_params.db_root_dir
    seq_name = spd_params.seq_name
    img_ext = spd_params.img_ext
    labels_ext = spd_params.labels_ext
    out_img_ext = spd_params.out_img_ext
    out_labels_ext = spd_params.out_labels_ext
    show_img = spd_params.vis
    patch_height = spd_params.patch_height
    patch_width = spd_params.patch_width
    min_stride = spd_params.min_stride
    max_stride = spd_params.max_stride
    enable_flip = spd_params.enable_flip
    enable_instance = spd_params.enable_instance
    min_rot = spd_params.min_rot
    max_rot = spd_params.max_rot
    n_frames = spd_params.n_frames
    start_id = spd_params.start_id
    end_id = spd_params.end_id
    sample = spd_params.sample
    shuffle = spd_params.shuffle
    dataset = batch_params.dataset.lower()

    if dataset in ['617', '617_val', 'ipsc']:
        """IPSC style labels with masks for each sequence in that sequence's image dir"""
        assert batch_params.labels_dir, "labels_dir must be provided"
        spd_params.src_path = linux_path(db_root_dir, seq_name)
        spd_params.labels_path = linux_path(db_root_dir, seq_name, batch_params.labels_dir)
        if batch_params.instances_dir:
            spd_params.instances_path = linux_path(db_root_dir, seq_name, batch_params.instances_dir)
    elif dataset == 'ctc':
        """CTC style labels with all masks in the same parent dir"""
        assert batch_params.image_dir, "image_dir must be provided"
        assert batch_params.labels_dir, "labels_dir must be provided"
        spd_params.src_path = linux_path(db_root_dir, batch_params.image_dir, seq_name)
        spd_params.labels_path = linux_path(db_root_dir, batch_params.labels_dir, seq_name)
    elif dataset == 'coco':
        """COCO style labels with all masks in the same parent dir"""
        assert batch_params.labels_dir, "labels_dir must be provided"
        spd_params.src_path = linux_path(db_root_dir, seq_name)
        spd_params.labels_path = linux_path(db_root_dir, batch_params.labels_dir, f'semantic_{seq_name}')
    elif dataset == 'cityscapes':
        """cityscapes style labels with all masks in the same parent dir under different city names"""
        assert batch_params.labels_dir, "labels_dir must be provided"
        spd_params.src_path = linux_path(db_root_dir, batch_params.image_dir, seq_name)
        spd_params.labels_path = linux_path(db_root_dir, batch_params.labels_dir, seq_name)
    else:
        raise AssertionError(f'invalid dataset: {batch_params.dataset}')

    src_files = [k for k in os.listdir(spd_params.src_path) if k.endswith('.{:s}'.format(img_ext))]
    total_frames = len(src_files)
    # print('file_list: {}'.format(file_list))
    assert total_frames > 0, 'No input frames found'

    # print('total_frames: {}'.format(total_frames))
    # print('n_frames: {}'.format(n_frames))

    if n_frames <= 0:
        n_frames = total_frames

    patch_mode = spd_params.patch_mode
    subseq_mode = spd_params.subseq_mode or start_id > 0

    if end_id < start_id:
        end_id = n_frames - 1
    else:
        subseq_mode = 1

    if spd_params.resize:
        spd_params.resize_x = spd_params.resize_y = spd_params.resize

    enable_resize = 0
    if spd_params.resize_x or spd_params.resize_y:
        enable_resize = 1

    if patch_height <= 0:
        if spd_params.resize_y:
            patch_height = spd_params.resize_y
    else:
        patch_mode = 1

    if patch_width <= 0:
        if spd_params.resize_x:
            patch_width = spd_params.resize_x
        else:
            patch_width = patch_height
    else:
        patch_mode = 1

    if min_stride <= 0:
        min_stride = patch_height
    else:
        patch_mode = 1

    if max_stride <= min_stride:
        max_stride = min_stride
    else:
        patch_mode = 1

    assert max_stride >= min_stride, "min_stride must be <= max_stride"

    if not patch_mode and enable_resize:
        if patch_height != spd_params.resize_y or patch_width != spd_params.resize_x:
            patch_mode = 1

    if not spd_params.out_suffix:
        out_suffixes = []
        if batch_params.split_suffix:
            out_suffixes.append(batch_params.split_suffix)

        if enable_resize:
            if spd_params.resize:
                out_suffixes.append(f'resize_{spd_params.resize}')
            else:
                out_suffixes.append(f'resize_{spd_params.resize_x}x{spd_params.resize_y}')

        if subseq_mode:
            out_suffixes += [
                f'{start_id:d}_{end_id:d}',
            ]

        if patch_mode:
            out_suffixes += [
                f'{patch_height:d}_{patch_width:d}',
                f'{min_stride:d}_{max_stride:d}',
            ]

        if shuffle:
            out_suffixes.append('rnd')

        if sample:
            out_suffixes.append('smp_{}'.format(sample))

        if batch_params.n_rot > 0:
            out_suffixes.append(f'rot_{min_rot:d}_{max_rot:d}_{batch_params.n_rot:d}')

        if enable_flip:
            out_suffixes.append('flip')

        if enable_instance:
            out_suffixes.append('inst')

        spd_params.out_suffix = '-'.join(out_suffixes)

    spd_params.db_root_dir = ''
    spd_params.seq_name = seq_name
    spd_params.img_ext = img_ext
    spd_params.labels_ext = labels_ext
    spd_params.out_img_ext = out_img_ext
    spd_params.out_labels_ext = out_labels_ext
    spd_params.patch_height = patch_height
    spd_params.patch_width = patch_width
    spd_params.min_stride = min_stride
    spd_params.max_stride = max_stride
    spd_params.enable_flip = enable_flip
    spd_params.start_id = start_id
    spd_params.end_id = end_id
    spd_params.n_frames = n_frames
    spd_params.vis = show_img

    _min_rot = min_rot

    for i in range(batch_params.n_rot):
        rot_range = int(float(max_rot - min_rot) / float(batch_params.n_rot))

        if i == batch_params.n_rot - 1:
            _max_rot = max_rot
        else:
            _max_rot = _min_rot + rot_range

        # rot_params = [
        #     f'enable_rot=1',
        #     f'min_rot={_min_rot}',
        #     f'max_rot={_max_rot}',
        # ]
        # rot_params_str = ' '.join(rot_params)
        # rot_cmd = f'{base_cmd} {rot_params_str}'
        # print(f'\n\nrot {i + 1} / {n_rot}:\n {rot_cmd}\n\n')
        # run_cmd(rot_cmd, params.parallel, params.log_to_file, params.log_dir, processes)

        spd_params.enable_rot = 1
        spd_params.min_rot = _min_rot
        spd_params.max_rot = _max_rot
        out_seq_name = f'{seq_name}-rot_{_min_rot}_{_max_rot}'
        set_output_paths(batch_params, spd_params, out_seq_name, db_root_dir)

        all_params.append(copy.deepcopy(spd_params))

        _min_rot = _max_rot + 1

    if min_rot > 0 or batch_params.n_rot == 0:
        """no rotation"""
        spd_params.enable_rot = 0
        set_output_paths(batch_params, spd_params, seq_name, db_root_dir)
        all_params.append(copy.deepcopy(spd_params))


def main():
    params = BatchParams()
    spd_params = spd.Params()

    spd_params.batch = params

    # paramparse.process(params, allow_unknown=True)
    paramparse.process(spd_params, allow_unknown=False)

    split = params.split
    dataset = params.dataset.lower()
    all_spd_params = []  # type: list[spd.Params]
    seq_suffix = None

    if dataset == '617':
        from db_info import RiverIceInfo as DBInfo
    elif dataset == '617_val':
        from db_info import RiverIceValInfo as DBInfo
    elif dataset == 'ipsc':
        from db_info import IPSCInfo as DBInfo
    elif dataset == 'ctc':
        from db_info import CTCInfo as DBInfo
    elif dataset == 'coco':
        from db_info import COCOInfo as DBInfo
    elif dataset in ['cityscapes', 'ctscp']:
        from db_info import CityscapesInfo as DBInfo
        if not os.path.isfile(spd_params.class_names_path):
            from cityscapesscripts.helpers.labels import labels
            class_txt_list = []
            for label in labels:
                if label.ignoreInEval:
                    continue
                if label.id < 0:
                    continue
                if label.id == 0:
                    """Cityscapes calls it unlabeled but the pipeline expects background"""
                    class_name = 'background'
                else:
                    class_name = label.name
                class_col_str = '_'.join(map(str, label.color[::-1]))
                class_txt = f'{class_name}\t{class_col_str}\t{label.id}'
                class_txt_list.append(class_txt)
            class_txt_all = '\n'.join(class_txt_list)
            with open(spd_params.class_names_path, 'w') as fid:
                fid.write(class_txt_all)
    else:
        raise AssertionError(f'unsupported dataset: {dataset}')

    db_splits = DBInfo.DBSplits().__dict__

    seq_ids = db_splits[split]

    if params.seq_id >= 0:
        params.seq_start_id = params.seq_end_id = params.seq_id

    if params.seq_start_id > 0 or params.seq_end_id >= 0:
        assert params.seq_end_id >= params.seq_start_id, "end_seq_id must to be >= start_seq_id"
        seq_suffix = f'seq_{params.seq_start_id}_{params.seq_end_id}'
        seq_ids = seq_ids[params.seq_start_id:params.seq_end_id + 1]

    n_seq = len(seq_ids)

    for __id, seq_id in enumerate(seq_ids):
        seq_name, n_frames = DBInfo.sequences[seq_id]
        spd_params.seq_name = seq_name
        get_spd_params(spd_params, params, all_spd_params)

    n_spd_params = len(all_spd_params)
    n_proc = params.n_proc
    if n_proc > n_spd_params:
        n_proc = n_spd_params

    image_infos = []

    if n_proc > 1:
        print(f'running {n_spd_params} configs in parallel over {n_proc} processes')
        import functools
        func = functools.partial(spd.run, image_infos)

        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(params.n_proc)

        # import multiprocessing
        # pool = multiprocessing.Pool(n_proc)

        pool.map(func, all_spd_params)
    else:
        for spd_id, spd_params_ in enumerate(all_spd_params):
            print(f'\n{spd_id + 1}/{n_spd_params}: {spd_params_.out_seq_name}')
            spd.run(image_infos, spd_params_)

    output_json_dict = {
        "images": image_infos,
        "type": "semantic",
        "annotations": [],
        "categories": [],
    }
    spd_params = all_spd_params[0]  # type: spd.Params

    assert spd_params.class_names_path, "class_names_path must be provided"

    class_id_to_col, class_id_to_name = task_utils.read_class_info(
        spd_params.class_names_path)

    for label_id, label in class_id_to_name.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    json_suffix = spd_params.out_suffix
    if seq_suffix is not None:
        json_suffix = f'{json_suffix}-{seq_suffix}'

    # if spd_params.enable_instance:
    #     json_suffix = f'{json_suffix}-inst'

    if spd_params.rle.json:
        if spd_params.rle.max_len > 0:
            json_suffix = f'{json_suffix}_max_{spd_params.rle.max_len}'
        if spd_params.rle.starts_2d:
            json_suffix = f'{json_suffix}_2d'

    output_json_fname = f'{json_suffix}.json'
    json_path = os.path.join(spd_params.out_root_dir, output_json_fname)
    save_json(output_json_dict, json_path)


if __name__ == '__main__':
    main()
