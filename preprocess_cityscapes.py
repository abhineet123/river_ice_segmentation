import shutil
import os
import glob
import cv2
import sys

p2s_path = os.path.join(os.path.expanduser("~"), "pix2seq")
sys.path.append(p2s_path)

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
sys.path.append(dproc_path)

import paramparse

from densenet.utils import linux_path, sort_key, col_bgr, resize_ar


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='ctscp')

        self.out_dir = ''
        self.root_dir = '/data/cityscapes'

        self.data_type = 'leftImg8bit'
        # self.data_type = 'rightImg8bit'

        self.gt_type = 'gtFine'
        # self.gt_type = 'gtCoarse'

        self.splits = []

        self.img_ext = 'png'

        self.gt_sub_types = [
            ('polygons', 'json'),
            ('color', 'png'),
            ('instanceIds', 'png'),
            ('labelIds', 'png'),
        ]


def reorganize_images_and_masks(params: Params):
    assert os.path.isdir(params.root_dir), "invalid root_dir"

    img_root_path = linux_path(params.root_dir, params.data_type)
    assert os.path.isdir(img_root_path), "invalid data_type"

    gt_root_path = linux_path(params.root_dir, params.gt_type)
    assert os.path.isdir(gt_root_path), "invalid gt_type"

    if not params.out_dir:
        params.out_dir = params.root_dir
        inplace = True
    else:
        inplace = False

    dst_img_root_path = linux_path(params.out_dir, params.data_type)
    os.makedirs(dst_img_root_path, exist_ok=True)

    dst_gt_root_paths = []
    for gt_sub_type, gt_ext in params.gt_sub_types:
        dst_gt_root_path = linux_path(params.out_dir, f'{params.gt_type}_{gt_sub_type}')
        os.makedirs(dst_gt_root_path, exist_ok=True)

        dst_gt_root_paths.append(dst_gt_root_path)

    splits = params.splits
    if not splits:
        splits = [k for k in os.listdir(img_root_path) if os.path.isdir(linux_path(img_root_path, k))]

    for split in splits:
        img_path = linux_path(img_root_path, split)
        gt_path = linux_path(gt_root_path, split)

        dst_img_path = linux_path(dst_img_root_path, split)
        os.makedirs(dst_img_path, exist_ok=True)

        cities = [k for k in os.listdir(img_path) if os.path.isdir(linux_path(img_path, k))]
        for city in cities:
            city_img_path = linux_path(img_path, city)
            city_gt_path = linux_path(gt_path, city)
            # img_files = [k for k in os.listdir(city_img_path) if os.path.isfile(linux_path(city_img_path, k))]
            img_files = glob.glob(f'{city_img_path}/*.{params.img_ext}')

            dst_city_img_path = linux_path(dst_img_path, city)

            os.makedirs(dst_city_img_path, exist_ok=True)

            gt_sub_info = []
            for (gt_sub_type, gt_ext), dst_gt_root_path in zip(params.gt_sub_types, dst_gt_root_paths, strict=True):
                dst_city_gt_path = linux_path(dst_gt_root_path, split, city)
                os.makedirs(dst_city_gt_path, exist_ok=True)

                gt_sub_info.append((gt_sub_type, gt_ext, dst_city_gt_path))

            for img_file in img_files:
                img_name, img_ext = os.path.splitext(os.path.basename(img_file))
                dst_img_name = img_name.replace(f'_{params.data_type}', '')
                dst_img_file = linux_path(dst_city_img_path, f'{dst_img_name}{img_ext}')

                if inplace:
                    shutil.move(img_file, dst_img_file)
                    # print(f'{img_file} --> {dst_img_file}')
                    # print(f'{dst_img_file}')

                    for gt_sub_type, gt_ext, dst_city_gt_path in gt_sub_info:
                        gt_file = linux_path(city_gt_path, f'{dst_img_name}_{params.gt_type}_{gt_sub_type}.{gt_ext}')
                        assert os.path.isfile(gt_file), f"non-nonexistent gt_file: {gt_file}"

                        dst_gt_file = linux_path(dst_city_gt_path, f'{dst_img_name}.{gt_ext}')

                        shutil.move(gt_file, dst_gt_file)
                        # print(f'\t{gt_file} --> {dst_gt_file}')
                        # print(f'\t{dst_gt_file}')
                    # print()


def main():
    params: Params = paramparse.process(Params)

    if params.data_type in ['leftImg8bit', 'rightImg8bit'] and params.gt_type in ['gtFine', 'gtCoarse']:
        reorganize_images_and_masks(params)


if __name__ == '__main__':
    main()
