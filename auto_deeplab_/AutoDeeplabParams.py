
class AutoDeeplabParams:
    """
    :param float arch_lr: learning rate for alpha and beta in architect searching process
    :param float arch_weight_decay: w-decay (default: 5e-4)
    :param str backbone: backbone name (default: resnet)
    :param int base_size: base image size
    :param int batch_size: input batch size for training (default: auto)
    :param str checkname: set the checkpoint name
    :param int crop_size: crop image size
    :param str dataset: dataset name (default: pascal)
    :param int epochs: number of epochs to train (default: auto)
    :param int eval_interval: evaluuation interval (default: 1)
    :param bool freeze_bn: whether to freeze bn parameters (default: False)
    :param bool ft: finetuning on a different dataset
    :param list gpu_ids: which GPU to train on (default: 0)
    :param str loss_type: loss func type (default: ce)
    :param float lr: learning rate (default: auto)
    :param str lr_scheduler: lr scheduler mode: (default: cos)
    :param float momentum: momentum (default: 0.9)
    :param bool nesterov: whether use nesterov (default: False)
    :param bool no_cuda: disables CUDA training
    :param bool no_val: skip validation during training
    :param int out_stride: network output stride (default: 8)
    :param int resize: resize image size
    :param str resume: put the path to resuming file if needed
    :param int seed: random seed (default: 1)
    :param int start_epoch: start epochs (default:0)
    :param bool sync_bn: whether to use sync bn (default: auto)
    :param int test_batch_size: input batch size for testing (default: auto)
    :param bool use_balanced_weights: whether to use balanced weights (default: False)
    :param bool use_sbd: whether to use SBD dataset (default: True)
    :param float weight_decay: w-decay (default: 5e-4)
    :param int workers: dataloader threads
    """

    def __init__(self):
        self.cfg = ('',)
        self.arch_lr = 0.003
        self.arch_weight_decay = 0.001
        self.backbone = 'resnet'
        self.base_size = 320
        self.batch_size = None
        self.checkname = None
        self.crop_size = 320
        self.dataset = 'pascal'
        self.epochs = None
        self.eval_interval = 1
        self.freeze_bn = False
        self.ft = False
        self.gpu_ids = [0, ]
        self.loss_type = 'ce'
        self.lr = 0.025
        self.lr_scheduler = 'cos'
        self.momentum = 0.9
        self.nesterov = False
        self.no_cuda = False
        self.no_val = False
        self.out_stride = 16
        self.resize = 512
        self.resume = None
        self.seed = 1
        self.start_epoch = 0
        self.sync_bn = None
        self.test_batch_size = None
        self.use_balanced_weights = False
        self.use_sbd = False
        self.weight_decay = 0.0003
        self.workers = 4
        self.help = {
        }
