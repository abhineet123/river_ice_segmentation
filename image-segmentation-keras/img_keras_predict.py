import argparse
import Models, LoadBatches
from keras.models import load_model
import glob, os, sys, time
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--test_images", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--n_classes", type=int)

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width = args.input_width
input_height = args.input_height
output_path = args.output_path
save_weights_path = args.save_weights_path

output_path_raw = os.path.join(output_path, 'raw/')

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

if not os.path.isfile(save_weights_path):
    save_weights_dir = os.path.dirname(save_weights_path)
    save_weights_name = os.path.basename(save_weights_path)

    weight_files = [k for k in os.listdir(save_weights_dir) if os.path.isfile(os.path.join(save_weights_dir, k))
                    and k.endswith('.h5') and k.startswith(save_weights_name)]
    if not weight_files:
        raise IOError('No weight files found matching {} in {}'.format(save_weights_name, save_weights_dir))
    weight_files.sort()
    save_weights_path = os.path.join(save_weights_dir, weight_files[-1])

sys.stdout.write('Loading weights from: {}\n'.format(save_weights_path))
sys.stdout.write('Saving results to: {}\n'.format(args.output_path))
sys.stdout.flush()

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(save_weights_path)
m.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth

print('Reading source images from: {}'.format(images_path))
images = glob.glob(images_path + "/*.jpg") + glob.glob(images_path + "/*.png") + glob.glob(images_path + "/*.jpeg")
images.sort()

n_images = len(images)
if n_images <= 0:
    raise SystemError('No input frames found')

# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]
colors = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]

if not os.path.isdir(output_path):
    os.makedirs(output_path)
if not os.path.isdir(output_path_raw):
    os.makedirs(output_path_raw)


sys.stdout.write('n_images: {}\n'.format(n_images))
sys.stdout.flush()

print('Saving raw segmentation images to : {}'.format(output_path_raw))

img_id = 0
for imgName in images:
    imgName_no_ext = os.path.splitext(os.path.basename(imgName))[0]
    outName = os.path.join(output_path, imgName_no_ext + '.png')
    outName_raw = os.path.join(output_path_raw, imgName_no_ext + '.png')

    _start_t = time.time()
    X = LoadBatches.getImageArr(imgName, args.input_width, args.input_height)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    _end_t = time.time()

    fps = 1.0 / float(_end_t - _start_t)

    # print('Writing to ', outName_raw)
    pr_resized = cv2.resize(pr.astype(np.uint8), (input_width, input_height))
    cv2.imwrite(outName_raw, pr_resized.astype(np.uint8))

    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)

    proc_end_t = time.time()
    proc_fps = 1.0 / float(proc_end_t - _start_t)

    sys.stdout.write('\rDone {:d}/{:d} images fps: {:.4f} ({:.4f})'.format(
        img_id + 1, n_images, fps, proc_fps))
    sys.stdout.flush()
    img_id += 1

sys.stdout.write('\n')
sys.stdout.flush()
