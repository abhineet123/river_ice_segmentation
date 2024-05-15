from paramparse import MultiPath
from new_deeplab.utils import linux_path

class NewDeeplabTrainParams:
    """
    :ivar logtostderr:  Should only log to stderr? (default: 'false')
    :type logtostderr: bool

    :ivar alsologtostderr:  also log to stderr? (default: 'false')
    :type alsologtostderr: bool

    :ivar v:
    :type v: int

    :ivar verbosity:
    :type verbosity: int

    :ivar stderrthreshold:  log messages at this level, or more severe, to stderr in addition to the logfile.
    Possible values are 'debug', 'info', 'warning', 'error', and 'fatal'.  Obsoletes --alsologtostderr. Using
    --alsologtostderr cancels the effect of this flag. Please also note that this flag is subject to --verbosity and
    requires logfile not be stderr. (default: 'fatal') -v,--verbosity: Logging verbosity level. Messages logged at
    this level or lower will be included. Set to 1 for debug logging. If the flag was not set or supplied,
    the value will be changed from the default of -1 (warning) to 0 (info) after flags are parsed. (default: '-1') (
    an integer)
    :type stderrthreshold: str

    :ivar showprefixforinfo:  If False, do not prepend prefix to info messages when it's logged to stderr,
    --verbosity is set to INFO level, and python logging is used. (default: 'true')
    :type showprefixforinfo: bool

    :ivar run_with_pdb:  Set to true for PDB debug mode (default: 'false')
    :type run_with_pdb: bool

    :ivar pdb_post_mortem:  Set to true to handle uncaught exceptions with PDB post mortem. (default: 'false')
    :type pdb_post_mortem: bool

    :ivar run_with_profiling:  Set to true for profiling the script. Execution will be slower, and the output format
    might change over time. (default: 'false')
    :type run_with_profiling: bool

    :ivar profile_file:  Dump profile information to a file (for python -m pstats). Implies --run_with_profiling.
    :type profile_file: str

    :ivar use_cprofile_for_profiling:  Use cProfile instead of the profile module for profiling. This has no effect
    unless --run_with_profiling is set. (default: 'true')
    :type use_cprofile_for_profiling: bool

    :ivar only_check_args:
    :type only_check_args: bool

    :ivar op_conversion_fallback_to_while_loop:
    :type op_conversion_fallback_to_while_loop: bool

    :ivar test_random_seed:  Random seed for testing. Some test frameworks may change the default value of this flag
    between runs, so it is not appropriate for seeding probabilistic tests. (default: '301') (an integer)
    :type test_random_seed: int

    :ivar test_srcdir:  Root of directory tree where source files live (default: '')
    :type test_srcdir: str

    :ivar test_tmpdir:  Directory for temporary testing files (default:
    'C:/Users/Tommy/AppData/Local/Temp/absl_testing')
    :type test_tmpdir: str

    :ivar test_randomize_ordering_seed:  If positive, use this as a seed to randomize the execution order for test
    cases. If "random", pick a random seed to use. If 0 or not set, do not randomize test case execution order. This
    flag also overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable.
    :type test_randomize_ordering_seed: NoneType

    :ivar xml_output_file:  File to store XML test results (default: '')
    :type xml_output_file: str

    :ivar num_clones:
    :type num_clones: int

    :ivar clone_on_cpu:  Use CPUs to deploy clones. (default: 'false')
    :type clone_on_cpu: bool

    :ivar num_replicas:
    :type num_replicas: int

    :ivar startup_delay_steps:  Number of training steps between replicas startup. (default: '15') (an integer)
    :type startup_delay_steps: int

    :ivar num_ps_tasks:
    :type num_ps_tasks: int

    :ivar master:  BNS name of the tensorflow server (default: '')
    :type master: str

    :ivar task:  The task ID. (default: '0') (an integer)
    :type task: int

    :ivar log_dir:  Where the checkpoint and logs are stored.
    :type log_dir: str

    :ivar log_steps:  Display logging information at every log_steps. (default: '10') (an integer)
    :type log_steps: int

    :ivar save_interval_secs:  How often, in seconds, we save the model to disk. (default: '1200') (an integer)
    :type save_interval_secs: int

    :ivar save_summaries_secs:  How often, in seconds, we compute the summaries. (default: '600') (an integer)
    :type save_summaries_secs: int

    :ivar save_summaries_images:  Save sample inputs, labels, and semantic predictions as images to summary. (
    default: 'false')
    :type save_summaries_images: bool

    :ivar profile_logdir:  Where the profile files are stored.
    :type profile_logdir: str

    :ivar learning_policy:  <poly|step>: Learning rate policy for training. (default: 'poly')
    :type learning_policy: str

    :ivar base_learning_rate:  The base learning rate for model training. (default: '0.0001') (a number)
    :type base_learning_rate: float

    :ivar learning_rate_decay_factor:  The rate to decay the base learning rate. (default: '0.1') (a number)
    :type learning_rate_decay_factor: float

    :ivar learning_rate_decay_step:  Decay the base learning rate at a fixed step. (default: '2000') (an integer)
    :type learning_rate_decay_step: int

    :ivar learning_power:  The power value used in the poly learning policy. (default: '0.9') (a number)
    :type learning_power: float

    :ivar train_steps:  The number of steps used for training (default: '30000') (an integer)
    :type train_steps: int

    :ivar momentum:  The momentum value to use (default: '0.9') (a number)
    :type momentum: float

    :ivar train_batch_size:  The number of images in each batch during training. (default: '8') (an integer)
    :type train_batch_size: int

    :ivar weight_decay:  The value of the weight decay for training. (default: '4e-05') (a number)
    :type weight_decay: float

    :ivar train_crop_size:  Image crop size [height, width] during training. (default: '513,513') (a comma separated
    list)
    :type train_crop_size: list

    :ivar last_layer_gradient_multiplier:  The gradient multiplier for last layers, which is used to boost the
    gradient of last layers if the value > 1. (default: '1.0') (a number)
    :type last_layer_gradient_multiplier: float

    :ivar upsample_logits:  Upsample logits during training. (default: 'true')
    :type upsample_logits: bool

    :ivar drop_path_keep_prob:  Probability to keep each path in the NAS cell when training. (default: '1.0') (a number)
    :type drop_path_keep_prob: float

    :ivar tf_initial_checkpoint:  The initial checkpoint in tensorflow format.
    :type tf_initial_checkpoint: str

    :ivar initialize_last_layer:  Initialize the last layer. (default: 'true')
    :type initialize_last_layer: bool

    :ivar last_layers_contain_logits_only:  Only consider logits as last layers or not. (default: 'false')
    :type last_layers_contain_logits_only: bool

    :ivar slow_start_step:  Training model with small learning rate for few steps. (default: '0') (an integer)
    :type slow_start_step: int

    :ivar slow_start_learning_rate:  Learning rate employed during slow start. (default: '0.0001') (a number)
    :type slow_start_learning_rate: float

    :ivar fine_tune_batch_norm:  Fine tune the batch norm parameters or not. (default: 'true')
    :type fine_tune_batch_norm: bool

    :ivar min_scale_factor:  Mininum scale factor for data augmentation. (default: '0.5') (a number)
    :type min_scale_factor: float

    :ivar max_scale_factor:  Maximum scale factor for data augmentation. (default: '2.0') (a number)
    :type max_scale_factor: float

    :ivar scale_factor_step_size:  Scale factor step size for data augmentation. (default: '0.25') (a number)
    :type scale_factor_step_size: float

    :ivar atrous_rates:  Atrous rates for atrous spatial pyramid pooling.; repeat this option to specify a list of
    values (an integer)

    :ivar output_stride:
    :type output_stride: int

    :ivar hard_example_mining_step:  The training step in which exact hard example mining kicks off. Note we
    gradually reduce the mining percent to the specified top_k_percent_pixels. For example,
    if hard_example_mining_step=100K and top_k_percent_pixels=0.25, then mining percent will gradually reduce from
    100%% to 25%% until 100K steps after which we only mine top 25%% pixels. (default: '0') (an integer)
    :type hard_example_mining_step: int

    :ivar top_k_percent_pixels:  The top k percent pixels (in terms of the loss values) used to compute loss during
    training. This is useful for hard pixel mining. (default: '1.0') (a number)
    :type top_k_percent_pixels: float

    :ivar quantize_delay_step:  Steps to start quantized training. If < 0, will not quantize model. (default: '-1') (
    an integer)
    :type quantize_delay_step: int

    :ivar dataset:  Name of the segmentation dataset. (default: 'pascal_voc_seg')
    :type dataset: str

    :ivar train_split:  Which split of the dataset to be used for training (default: 'train')

    :ivar dataset_dir:  Where the dataset reside.
    :type dataset_dir: str

    :ivar allow_memory_growth:  allow_memory_growth (default: '1') (an integer)
    :type allow_memory_growth: int

    :ivar gpu_memory_fraction:  gpu_memory_fraction. (default: '1.0') (a number)
    :type gpu_memory_fraction: float

    :ivar min_resize_value:  Desired size of the smaller image side. (an integer)
    :type min_resize_value: int

    :ivar max_resize_value:  Maximum allowed size of the larger image side. (an integer)
    :type max_resize_value: int

    :ivar resize_factor:  Resized dimensions are multiple of factor plus one. (an integer)
    :type resize_factor: int

    :ivar logits_kernel_size:  The kernel size for the convolutional kernel that generates logits. (default: '1') (an
    integer)
    :type logits_kernel_size: int

    :ivar model_variant:  DeepLab model variant. (default: 'mobilenet_v2')
    :type model_variant: str

    :ivar image_pyramid:  Input scales for multi-scale feature extraction.; repeat this option to specify a list of
    values (a number)

    :ivar add_image_level_feature:  Add image level feature. (default: 'true')
    :type add_image_level_feature: bool

    :ivar image_pooling_crop_size:  Image pooling crop size [height, width] used in the ASPP module. When value is
    None, the model performs image pooling with "crop_size". Thisflag is useful when one likes to use different image
    pooling sizes. (a comma separated list)
    :type image_pooling_crop_size: list

    :ivar image_pooling_stride:  Image pooling stride [height, width] used in the ASPP image pooling. (default: '1,
    1') (a comma separated list)
    :type image_pooling_stride: list

    :ivar aspp_with_batch_norm:  Use batch norm parameters for ASPP or not. (default: 'true')
    :type aspp_with_batch_norm: bool

    :ivar aspp_with_separable_conv:  Use separable convolution for ASPP or not. (default: 'true')
    :type aspp_with_separable_conv: bool

    :ivar multi_grid:  Employ a hierarchy of atrous rates for ResNet.; repeat this option to specify a list of values
    (an integer)

    :ivar depth_multiplier:  Multiplier for the depth (number of channels) for all convolution ops used in MobileNet.
    (default: '1.0') (a number)
    :type depth_multiplier: float

    :ivar divisible_by:  An integer that ensures the layer # channels are divisible by this value. Used in MobileNet.
    (an integer)
    :type divisible_by: int

    :ivar decoder_output_stride:  Comma-separated list of strings with the number specifying output stride of
    low-level features at each network level.Current semantic segmentation implementation assumes at most one output
    stride (i.e., either None or a list with only one element. (a comma separated list)
    :type decoder_output_stride: list

    :ivar decoder_use_separable_conv:  Employ separable convolution for decoder or not. (default: 'true')
    :type decoder_use_separable_conv: bool

    :ivar merge_method:  <max|avg>: Scheme to merge multi scale features. (default: 'max')
    :type merge_method: str

    :ivar prediction_with_upsampled_logits:  When performing prediction, there are two options: (1) bilinear
    upsampling the logits followed by argmax, or (2) armax followed by nearest upsampling the predicted labels. The
    second option may introduce some "blocking effect", but it is more computationally efficient. Currently,
    prediction_with_upsampled_logits=False is only supported for single-scale inference. (default: 'true')
    :type prediction_with_upsampled_logits: bool

    :ivar dense_prediction_cell_json:  A JSON file that specifies the dense prediction cell. (default: '')
    :type dense_prediction_cell_json: str

    :ivar nas_stem_output_num_conv_filters:
    :type nas_stem_output_num_conv_filters: int

    :ivar use_bounded_activation:  Whether or not to use bounded activations. Bounded activations better lend
    themselves to quantized inference. (default: 'false')
    :type use_bounded_activation: bool

    """

    def __init__(self):
        self.logtostderr = False
        self.alsologtostderr = False

        self.stderrthreshold = 'fatal'
        self.showprefixforinfo = True

        self.v = -1
        self.verbosity = -1
        self.run_with_pdb = False
        self.pdb_post_mortem = False
        self.run_with_profiling = False
        self.profile_file = None
        self.use_cprofile_for_profiling = True
        self.only_check_args = False
        self.op_conversion_fallback_to_while_loop = False
        self.test_random_seed = 301
        self.test_randomize_ordering_seed = 0
        self.xml_output_file = ''
        self.num_clones = 1
        self.clone_on_cpu = False
        self.num_replicas = 1
        self.startup_delay_steps = 15
        self.num_ps_tasks = 0
        self.master = ''
        self.task = 0

        self.log_steps = 10
        self.save_interval_secs = 12
        self.save_summaries_secs = 6
        self.save_summaries_images = True
        self.profile_logdir = None
        self.learning_policy = 'poly'
        self.base_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.1
        self.learning_rate_decay_step = 2000
        self.learning_power = 0.9
        self.train_steps = 5000
        self.momentum = 0.9
        self.weight_decay = 4e-05
        self.train_crop_size = [513, 513]
        self.last_layer_gradient_multiplier = 1.0
        self.upsample_logits = True
        self.drop_path_keep_prob = 1.0
        self.tf_initial_checkpoint = None
        self.initialize_last_layer = True
        self.last_layers_contain_logits_only = False
        self.slow_start_step = 0
        self.slow_start_learning_rate = 0.0001
        self.fine_tune_batch_norm = True
        self.min_scale_factor = 0.5
        self.max_scale_factor = 2.0
        self.scale_factor_step_size = 0.25
        self.atrous_rates = [6, 12, 18]
        self.output_stride = 16
        self.hard_example_mining_step = 0
        self.top_k_percent_pixels = 1.0
        self.quantize_delay_step = -1

        self.allow_memory_growth = 1
        self.gpu_memory_fraction = 1.0
        self.min_resize_value = None
        self.max_resize_value = None
        self.resize_factor = None
        self.logits_kernel_size = 1
        self.model_variant = 'mobilenet_v2'
        self.image_pyramid = []
        self.add_image_level_feature = True
        self.image_pooling_crop_size = None
        self.image_pooling_stride = [1, 1]
        self.aspp_with_batch_norm = True
        self.aspp_with_separable_conv = True
        self.multi_grid = None
        self.depth_multiplier = 1.0
        self.divisible_by = None
        self.decoder_output_stride = None
        self.decoder_use_separable_conv = True
        self.merge_method = 'max'
        self.prediction_with_upsampled_logits = True
        self.dense_prediction_cell_json = ''
        self.nas_stem_output_num_conv_filters = 20
        self.use_bounded_activation = False

        self.train_batch_size = 8

        self.db_root_dir = '/data'

        self.dataset = ''
        self.dataset_dir = ''

        self.model_info = MultiPath()
        self.train_info = MultiPath()
        self.train_split = MultiPath()

        self.log_dir = ''
        self.tb_dir = ''
        self.checkpoint_dir = ''

        self.class_info_path = 'data/classes_ice.txt'


    def process(self):
        if not self.dataset_dir:
            self.dataset_dir = linux_path(self.db_root_dir, 'tfrecord', self.dataset, 'train')

        if not self.log_dir:
            self.log_dir = linux_path('log', self.train_info, self.model_info)

        if not self.tb_dir:
            self.tb_dir = linux_path(self.log_dir, 'tb')

        if not self.checkpoint_dir:
            self.checkpoint_dir = linux_path('ckpt', self.train_info, self.model_info)
