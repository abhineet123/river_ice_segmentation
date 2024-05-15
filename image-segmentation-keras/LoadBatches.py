import numpy as np
import cv2
import glob, sys
import itertools
from pprint import pprint


def getImageArr(path, width, height, imgNorm="sub_mean", ordering='channels_first'):
    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0

        if ordering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        if ordering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img


def getSegmentationArr(path, nClasses, width, height, _idx=None):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))

    if _idx is not None:
        # seg_labels = seg_labels[_idx, :]
        seg_labels[_idx, :] = -1

    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width, selective_loss=0, ordering='channels_first', start_id=0, end_id=-1, debug_mode=0):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    # print('images: {}'.format(images))
    # print('segmentations: {}'.format(segmentations))

    if end_id < start_id:
        end_id = len(images) - 1

    images = images[start_id:end_id + 1]
    segmentations = segmentations[start_id:end_id + 1]

    # print('images: {}'.format(images))
    # print('segmentations: {}'.format(segmentations))

    if selective_loss:
        print('Discarding images with less than {} pixels in each class...'.format(selective_loss))
        n_images = len(images)
        _images = []
        _segmentations = []
        label_masks = []
        # label_indices = []
        n_skipped = 0
        for img_id in range(n_images):
            labels_img_fname = segmentations[img_id]
            labels_img = cv2.imread(labels_img_fname)

            labels_img = cv2.resize(labels_img, (output_width, output_height))

            if labels_img is None:
                raise SystemError('Labels image could not be read from: {}'.format(labels_img_fname))

            if len(labels_img.shape) == 3:
                labels_img = labels_img[:, :, 0].squeeze()

            curr_labels_indices = None

            # Y = np.zeros((height * width, n_classes), dtype=np.float32)
            skip_image = 0
            for class_id in range(n_classes):
                class_indices = np.flatnonzero(labels_img == class_id)
                if class_indices.shape[0] < selective_loss:
                    skip_image = 1
                    # print(
                    #     '\nimg {} class {} class_indices.shape: {} '.format(img_id + 1, class_id, class_indices.shape))
                    break
                class_indices = np.random.choice(class_indices, (selective_loss, 1), replace=False).squeeze()
                # print('class_indices.shape: {}'.format(class_indices.shape))
                if curr_labels_indices is None:
                    curr_labels_indices = class_indices
                else:
                    curr_labels_indices = np.concatenate((curr_labels_indices, class_indices), axis=0)

            if skip_image:
                n_skipped += 1
                continue

            # print('len(curr_labels_indices): {}'.format(len(curr_labels_indices)))

            # curr_labels_indices = np.array(curr_labels_indices, dtype=np.int32).flatten()
            # print('curr_labels_indices.shape: {}'.format(curr_labels_indices.shape))

            h, w = labels_img.shape
            mask = np.ones(h * w, np.bool)
            mask[curr_labels_indices] = 0
            # print('mask.shape: {}'.format(mask.shape))
            label_masks.append(mask)

            # print('labels_img.shape: {}'.format(labels_img.flatten().shape))
            # print('curr_labels_indices.shape: {}'.format(curr_labels_indices.shape))

            # label_indices.append(curr_labels_indices)
            src_img_fname = images[img_id]

            _images.append(src_img_fname)
            _segmentations.append(labels_img_fname)

            sys.stdout.write('\rDone {}/{} images ({} skipped)'.format(img_id + 1, n_images, n_skipped))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        images = _images
        segmentations = _segmentations

        print('images:')
        pprint(images)
        print('segmentations:')
        pprint(segmentations)
        zipped = itertools.cycle(zip(images, segmentations, label_masks))

    else:
        zipped = itertools.cycle(zip(images, segmentations))

    idx = None
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            if selective_loss:
                im, seg, idx = zipped.next()
            else:
                im, seg = zipped.next()

            X.append(getImageArr(im, input_width, input_height, ordering=ordering))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height, idx))

            # valid_idx = np.count_nonzero(np.logical_not(idx))
            # valid_labels = np.count_nonzero(Y[-1] >= 0)

            # if debug_mode:
            #     print('idx.size: {}'.format(idx.size))
            #     print('valid_idx: {}'.format(valid_idx))
            #     print('Y[-1].size: {}'.format(Y[-1].size))
            #     print('valid_labels: {}'.format(valid_labels))

        yield np.array(X), np.array(Y)
        # yield ({'img_input': np.array(X), 'indices_input': np.array(idx)}, {'output': np.array(Y)})
        # yield ({'img_input': np.array(X), 'indices_input': np.array(idx)})

# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )
