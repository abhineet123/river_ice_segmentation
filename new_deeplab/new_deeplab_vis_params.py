from paramparse import MultiPath
from new_deeplab.utils import linux_path


class NewDeeplabVisParams:
    """
    :ivar add_flipped_images:  Add flipped images for evaluation or not. (default: 'false')
    :type add_flipped_images: bool

    :ivar add_image_level_feature:  Add image level feature. (default: 'true')
    :type add_image_level_feature: bool

    :ivar also_save_raw_predictions:  Also save raw predictions. (default: 'true')
    :type also_save_raw_predictions: bool

    :ivar also_save_vis_predictions:  Also save visualization predictions. (default: '0') (an integer)
    :type also_save_vis_predictions: int

    :ivar alsologtostderr:  also log to stderr? (default: 'false')
    :type alsologtostderr: bool

    :ivar aspp_with_batch_norm:  Use batch norm parameters for ASPP or not. (default: 'true')
    :type aspp_with_batch_norm: bool

    :ivar aspp_with_separable_conv:  Use separable convolution for ASPP or not. (default: 'true')
    :type aspp_with_separable_conv: bool

    :ivar atrous_rates:  Atrous rates for atrous spatial pyramid pooling.; repeat this option to specify a list of
    values (an integer)
    :type atrous_rates: list

    :ivar colormap_type:  <pascal|cityscapes>: Visualization colormap type. (default: 'pascal')
    :type colormap_type: str

    :ivar dataset:  Name of the segmentation dataset. (default: 'pascal_voc_seg')
    :type dataset: str

    :ivar dataset_dir:  Where the dataset reside.
    :type dataset_dir: str

    :ivar decoder_output_stride:  Comma-separated list of strings with the number specifying output stride of
    low-level features at each network level.Current semantic segmentation implementation assumes at most one output
    stride (i.e., either None or a list with only one element. (a comma separated list)
    :type decoder_output_stride: list

    :ivar decoder_use_separable_conv:  Employ separable convolution for decoder or not. (default: 'true')
    :type decoder_use_separable_conv: bool

    :ivar dense_prediction_cell_json:  A JSON file that specifies the dense prediction cell. (default: '')
    :type dense_prediction_cell_json: str

    :ivar depth_multiplier:  Multiplier for the depth (number of channels) for all convolution ops used in MobileNet.
    (default: '1.0') (a number)
    :type depth_multiplier: float

    :ivar divisible_by:  An integer that ensures the layer # channels are divisible by this value. Used in MobileNet.
    (an integer)
    :type divisible_by: int

    :ivar eval_interval_secs:  How often (in seconds) to run evaluation. (default: '300') (an integer)
    :type eval_interval_secs: int

    :ivar eval_scales:  The scales to resize images for evaluation.; repeat this option to specify a list of values (
    default: '[1.0]') (a number)
    :type eval_scales: list

    :ivar image_pooling_crop_size:  Image pooling crop size [height, width] used in the ASPP module. When value is
    None, the model performs image pooling with "crop_size". Thisflag is useful when one likes to use different image
    pooling sizes. (a comma separated list)
    :type image_pooling_crop_size: list

    :ivar image_pooling_stride:  Image pooling stride [height, width] used in the ASPP image pooling. (default: '1,
    1') (a comma separated list)
    :type image_pooling_stride: list

    :ivar image_pyramid:  Input scales for multi-scale feature extraction.; repeat this option to specify a list of
    values (a number)
    :type image_pyramid: list

    :ivar log_dir:  directory to write logfiles into (default: '')
    :type log_dir: str

    :ivar logits_kernel_size:  The kernel size for the convolutional kernel that generates logits. (default: '1') (an
    integer)
    :type logits_kernel_size: int

    :ivar logtostderr:  Should only log to stderr? (default: 'false')
    :type logtostderr: bool

    :ivar master:  BNS name of the tensorflow server (default: '')
    :type master: str

    :ivar max_number_of_iterations:  Maximum number of visualization iterations. Will loop indefinitely upon
    nonpositive values. (default: '0') (an integer)
    :type max_number_of_iterations: int

    :ivar max_resize_value:  Maximum allowed size of the larger image side. (an integer)
    :type max_resize_value: int

    :ivar merge_method:  <max|avg>: Scheme to merge multi scale features. (default: 'max')
    :type merge_method: str

    :ivar min_resize_value:  Desired size of the smaller image side. (an integer)
    :type min_resize_value: int

    :ivar model_variant:  DeepLab model variant. (default: 'mobilenet_v2')
    :type model_variant: str

    :ivar multi_grid:  Employ a hierarchy of atrous rates for ResNet.; repeat this option to specify a list of values
    (an integer)
    :type multi_grid: list

    :ivar nas_stem_output_num_conv_filters:
    :type nas_stem_output_num_conv_filters: int

    :ivar only_check_args:
    :type only_check_args: bool

    :ivar op_conversion_fallback_to_while_loop:
    :type op_conversion_fallback_to_while_loop: bool

    :ivar output_stride:
    :type output_stride: int

    :ivar pdb_post_mortem:  Set to true to handle uncaught exceptions with PDB post mortem. (default: 'false')
    :type pdb_post_mortem: bool

    :ivar prediction_with_upsampled_logits:  When performing prediction, there are two options: (1) bilinear
    upsampling the logits followed by argmax, or (2) armax followed by nearest upsampling the predicted labels. The
    second option may introduce some "blocking effect", but it is more computationally efficient. Currently,
    prediction_with_upsampled_logits=False is only supported for single-scale inference. (default: 'true')
    :type prediction_with_upsampled_logits: bool

    :ivar profile_file:  Dump profile information to a file (for python -m pstats). Implies --run_with_profiling.
    :type profile_file: int

    :ivar quantize_delay_step:  Steps to start quantized training. If < 0, will not quantize model. (default: '-1') (
    an integer)
    :type quantize_delay_step: int

    :ivar resize_factor:  Resized dimensions are multiple of factor plus one. (an integer)
    :type resize_factor: int

    :ivar run_with_pdb:  Set to true for PDB debug mode (default: 'false')
    :type run_with_pdb: bool

    :ivar run_with_profiling:  Set to true for profiling the script. Execution will be slower, and the output format
    might change over time. (default: 'false')
    :type run_with_profiling: bool

    :ivar showprefixforinfo:  If False, do not prepend prefix to info messages when it's logged to stderr,
    --verbosity is set to INFO level, and python logging is used. (default: 'true')
    :type showprefixforinfo: bool

    :ivar stderrthreshold:  log messages at this level, or more severe, to stderr in addition to the logfile.
    Possible values are 'debug', 'info', 'warning', 'error', and 'fatal'.  Obsoletes --alsologtostderr. Using
    --alsologtostderr cancels the effect of this flag. Please also note that this flag is subject to --verbosity and
    requires logfile not be stderr. (default: 'fatal') -v,--verbosity: Logging verbosity level. Messages logged at
    this level or lower will be included. Set to 1 for debug logging. If the flag was not set or supplied,
    the value will be changed from the default of -1 (warning) to 0 (info) after flags are parsed. (default: '-1') (
    an integer)
    :type stderrthreshold: str

    :ivar test_random_seed:  Random seed for testing. Some test frameworks may change the default value of this flag
    between runs, so it is not appropriate for seeding probabilistic tests. (default: '301') (an integer)
    :type test_random_seed: int

    :ivar test_randomize_ordering_seed:  If positive, use this as a seed to randomize the execution order for test
    cases. If "random", pick a random seed to use. If 0 or not set, do not randomize test case execution order. This
    flag also overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable.
    :type test_randomize_ordering_seed: str

    :ivar test_srcdir:  Root of directory tree where source files live (default: '')
    :type test_srcdir: str

    :ivar test_tmpdir:  Directory for temporary testing files (default:
    'C:\\Users\\Tommy\\AppData\\Local\\Temp\\absl_testing')
    :type test_tmpdir: str

    :ivar use_bounded_activation:  Whether or not to use bounded activations. Bounded activations better lend
    themselves to quantized inference. (default: 'false')
    :type use_bounded_activation: bool

    :ivar use_cprofile_for_profiling:  Use cProfile instead of the profile module for profiling. This has no effect
    unless --run_with_profiling is set. (default: 'true')
    :type use_cprofile_for_profiling: bool

    :ivar v:
    :type v: int

    :ivar verbosity:
    :type verbosity: int

    :ivar vis_batch_size:  The number of images in each batch during evaluation. (default: '1') (an integer)
    :type vis_batch_size: int

    :ivar vis_crop_size:  Crop size [height, width] for visualization. (default: '513,513') (a comma separated list)
    :type vis_crop_size: list

    :ivar train_split:  Which split of the dataset used for visualizing results (default: 'val')

    :ivar xml_output_file:  File to store XML test results (default: '')
    :type xml_output_file: str

    """

    def __init__(self):
        self.cfg = ()
        self.add_flipped_images = False
        self.add_image_level_feature = False
        self.also_save_raw_predictions = True
        self.also_save_vis_predictions = 0

        self.alsologtostderr = False

        self.aspp_with_batch_norm = True
        self.aspp_with_separable_conv = True
        self.atrous_rates = []
        self.colormap_type = 'pascal'

        self.decoder_output_stride = None
        self.decoder_use_separable_conv = True
        self.dense_prediction_cell_json = ''
        self.depth_multiplier = 1.0
        self.divisible_by = None
        self.eval_interval_secs = 0
        self.eval_scales = [1.0, ]
        self.image_pooling_crop_size = None
        self.image_pooling_stride = [1, 1]
        self.image_pyramid = None
        self.logits_kernel_size = 1

        self.logtostderr = False

        self.master = ''
        self.max_number_of_iterations = 1
        self.max_resize_value = None
        self.merge_method = 'max'
        self.min_resize_value = None
        self.model_variant = 'mobilenet_v2'
        self.multi_grid = None
        self.nas_stem_output_num_conv_filters = 20
        self.only_check_args = False
        self.op_conversion_fallback_to_while_loop = False
        self.output_stride = 16
        self.pdb_post_mortem = False
        self.prediction_with_upsampled_logits = True
        self.profile_file = None
        self.quantize_delay_step = -1
        self.resize_factor = None
        self.run_with_pdb = False
        self.run_with_profiling = False

        self.showprefixforinfo = True
        self.stderrthreshold = 'fatal'

        self.test_random_seed = 301
        self.test_randomize_ordering_seed = None
        self.test_srcdir = ''

        self.use_bounded_activation = False
        self.use_cprofile_for_profiling = True
        self.v = -1
        self.verbosity = -1
        self.vis_batch_size = 1
        self.vis_crop_size = [513, 513]

        self.model_info = MultiPath()
        self.train_info = MultiPath()
        self.train_split = MultiPath()
        self.vis_info = MultiPath()
        self.vis_split = MultiPath()

        self.dataset = ''
        self.vis_type = 'test'
        self.dataset_dir = ''
        self.db_root_dir = '/data'
        self.xml_output_file = ''

        self.log_dir = ''
        self.checkpoint_dir = ''
        self.vis_logdir = ''

        self.class_info_path = 'data/classes_ice.txt'


    def process(self):
        if not self.dataset_dir:
            self.dataset_dir = linux_path(self.db_root_dir, 'tfrecord', self.dataset, self.vis_type)

        if not self.log_dir:
            self.log_dir = linux_path('log', self.train_info, self.model_info)

        if not self.checkpoint_dir:
            self.checkpoint_dir = linux_path('ckpt', self.train_info, self.model_info)

        if not self.vis_logdir:
            self.vis_logdir = linux_path(self.log_dir, self.vis_info)
