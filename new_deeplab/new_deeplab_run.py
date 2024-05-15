import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys

sys.path.append('..')

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

import paramparse

import sys

sys.path.append('..')

from new_deeplab_train_params import NewDeeplabTrainParams
from new_deeplab_vis_params import NewDeeplabVisParams
#
# from visDataset import VisParams
# from stitchSubPatchDataset import StitchParams

import new_deeplab_train as train
import new_deeplab_vis as raw_vis

import stitchSubPatchDataset as stitch
import visDataset as vis


class Phases:
    train, raw_vis, stitch, vis = map(str, range(4))


class NewDeeplabParams:
    def __init__(self):
        self.cfg_root = 'cfg'
        self.cfg_ext = 'cfg'
        self.cfg = ()

        self.gpu = ""

        self.phases = '013'
        self.start = 0

        self.train = NewDeeplabTrainParams()
        self.raw_vis = NewDeeplabVisParams()
        self.stitch = stitch.Params()
        self.vis = vis.Params()

    def process(self):
        self.train.process()
        self.raw_vis.process()
        self.stitch.process()
        self.vis.process()


def main():
    params = NewDeeplabParams()
    paramparse.process(params)

    params.process()

    phases = params.phases
    if params.start > 0:
        phases = phases[params.start:]

    if params.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    if Phases.train in phases:
        train.run(params.train)

    if Phases.raw_vis in phases:
        raw_vis.run(params.raw_vis)

    if Phases.stitch in phases:
        stitch.run(params.stitch)

    if Phases.vis in phases:
        vis.run(params.vis)


if __name__ == '__main__':
    main()
