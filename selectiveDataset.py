import os
import sys
import numpy as np
from densenet.utils import processArguments, read_data
from imageio import imread, imsave

params = {
    'db_root_dir': '/home/abhineet/N/Datasets/617/',
    'src_dir': '',
    'dst_dir': '',
    'images_ext': 'png',
    'labels_ext': 'png',
    'n_classes': 3,
    'n_indices': 0,
    'start_id': 0,
    'end_id': -1,
    'copy_images': 0,
}

if __name__ == '__main__':
    processArguments(sys.argv[1:], params)
    db_root_dir = params['db_root_dir']
    src_dir = params['src_dir']
    dst_dir = params['dst_dir']
    images_ext = params['images_ext']
    labels_ext = params['labels_ext']
    n_classes = params['n_classes']
    n_indices = params['n_indices']
    start_id = params['start_id']
    end_id = params['end_id']
    copy_images = params['copy_images']

    images_path = os.path.join(db_root_dir, src_dir, 'images')
    labels_path = os.path.join(db_root_dir, src_dir, 'labels')

    images_path = os.path.abspath(images_path)
    labels_path = os.path.abspath(labels_path)

    src_files, src_labels_list, total_frames = read_data(images_path, images_ext, labels_path,
                                                         labels_ext)
    if start_id < 0:
        if end_id < 0:
            raise AssertionError('end_id must be non negative for random selection')
        elif end_id >= total_frames:
            raise AssertionError('end_id must be less than total_frames for random selection')
        print('Using {} random images for selection'.format(end_id + 1))
        img_ids = np.random.choice(total_frames, end_id + 1, replace=False)
    else:
        if end_id < start_id:
            end_id = total_frames - 1
        print('Using all {} images for selection'.format(end_id - start_id + 1))
        img_ids = range(start_id, end_id + 1)

    if not dst_dir:
        dst_dir = '{}_sel_{}'.format(src_dir, n_indices)

    out_dir = os.path.join(db_root_dir, dst_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_images_path = os.path.join(out_dir, 'images')
    out_labels_path = os.path.join(out_dir, 'labels')

    if not os.path.isdir(out_images_path):
        os.makedirs(out_images_path)

    if not os.path.isdir(out_labels_path):
        os.makedirs(out_labels_path)

    print('Saving output images to {} and labels to {}'.format(out_images_path, out_labels_path))

    _n_images = len(img_ids)
    n_skipped = 0

    for _id, img_id in enumerate(img_ids):
        img_fname = src_files[img_id]
        img_fname_no_ext = os.path.splitext(img_fname)[0]

        labels_img_fname = os.path.join(labels_path, img_fname_no_ext + '.{}'.format(labels_ext))
        labels_img = imread(labels_img_fname)

        if labels_img is None:
            raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

        if len(labels_img.shape) == 3:
            labels_img = labels_img[:, :, 0].squeeze()

        labels_indices = []

        curr_labels_indices = None

        skip_image = 0
        for class_id in range(n_classes):
            class_indices = np.flatnonzero(labels_img == class_id)
            if n_indices > 0:
                if class_indices.shape[0] < n_indices:
                    skip_image = 1
                    break
                class_indices = np.random.choice(class_indices, (n_indices, 1), replace=False).squeeze()
            if curr_labels_indices is None:
                curr_labels_indices = class_indices
            else:
                curr_labels_indices = np.concatenate((curr_labels_indices, class_indices), axis=0)

        if skip_image:
            continue

        mask = np.ones(labels_img.shape, np.bool)
        mask[np.unravel_index(curr_labels_indices, labels_img.shape)] = 0

        labels_img[mask] = 255

        dst_labels_img_fname = os.path.join(out_labels_path, img_fname)
        imsave(dst_labels_img_fname, labels_img)

        src_img_fname = os.path.join(images_path, img_fname)
        dst_img_fname = os.path.join(out_images_path, img_fname)

        if copy_images:
            os.system('cp {} {}'.format(src_img_fname, dst_img_fname))
        else:
            os.system('ln -s {} {}'.format(src_img_fname, dst_img_fname))

        sys.stdout.write('\rDone {}/{} images ({} skipped)'.format(_id + 1, _n_images, n_skipped))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
