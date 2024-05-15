import numpy as np
import cv2

# BGR values for different colors
col_bgr = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}


def read_class_info(class_info_path):
    is_composite = 0
    class_info = []
    composite_class_info = []
    class_lines = [k.strip() for k in open(class_info_path, 'r').readlines()]

    classes = []

    for line in class_lines:
        if not line:
            is_composite = 1
            continue

        if is_composite:
            _class, _class_col, _base_classes = line.split('\t')
            _base_classes = _base_classes.split(',')

            assert len(_base_classes) >= 2, "composite class must have at least 2 base classes"

            _base_class_ids = []
            for _base_class in _base_classes:
                try:
                    _base_class_id = classes.index(_base_class)
                except IndexError:
                    raise AssertionError("invalid base_class: {} for composite_class: {}".format(
                        _base_class, _class))
                _base_class_ids.append(_base_class_id)
            composite_class_info.append((_class, col_bgr[_class_col], _base_class_ids))
        else:
            _class, _class_col = line.split('\t')

            class_info.append((_class, col_bgr[_class_col]))
            classes.append(_class)

    return class_info, composite_class_info


def undo_resize_ar(resized_img, src_width, src_height, placement_type=0):
    height, width = resized_img.shape[:2]
    src_aspect_ratio = float(src_width) / float(src_height)
    aspect_ratio = float(width) / float(height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type == 0:
            start_row = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type == 0:
            start_col = 0
        elif placement_type == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    height_resize_factor = float(dst_height) / float(height)
    width_resize_factor = float(dst_width) / float(width)

    # assert height_resize_factor == width_resize_factor, "mismatch between height and width resize_factors"

    resized_img = resized_img.astype(np.uint8)
    unscaled_img = cv2.resize(resized_img, (dst_width, dst_height))
    unpadded_img = unscaled_img[start_row:start_row + src_height, start_col:start_col + src_width, ...]

    unpadded_img_disp, _, _ = resize_ar(unpadded_img, 640)

    # print('width, height: {}'.format((width, height)))
    # print('dst_width, dst_height: {}'.format((dst_width, dst_height)))
    # print('src_width, src_height: {}'.format((src_width, src_height)))

    # cv2.imshow('resized_img', resized_img)
    # cv2.imshow('unpadded_img', unpadded_img_disp)
    # cv2.waitKey(0)

    return unpadded_img


def raw_seg_to_rgb(raw_seg_img, class_to_color):
    seg_img = raw_seg_img.copy()
    if len(seg_img.shape) != 3:
        seg_img = np.stack((seg_img, seg_img, seg_img), axis=2)
    else:
        raw_seg_img = raw_seg_img[..., :0].squeeze()

    for _id, _col in class_to_color.items():
        seg_img[raw_seg_img == _id] = _col

    return seg_img


def remove_fuzziness_in_mask(seg_img, n_classes, class_to_color, fuzziness, check_equality=1):
    """handle annoying nearby pixel values to each actual class label, e.g. 253, 254 for actual label 255"""

    seg_img_min = np.amin(seg_img)
    seg_img_max = np.amax(seg_img)

    seg_img_out = np.copy(seg_img)
    seg_img_raw = np.zeros(seg_img.shape[:2], dtype=np.uint8)

    class_to_ids = {}

    for _id in range(n_classes):
        actual_col = class_to_color[_id]

        fuzzy_range = [(max(0, k - fuzziness), min(255, k + fuzziness)) for k in actual_col]
        fuzzy_range_red, fuzzy_range_green, fuzzy_range_blue = fuzzy_range

        seg_img_red = seg_img[..., 0]
        seg_img_green = seg_img[..., 1]
        seg_img_blue = seg_img[..., 2]

        fuzzy_ids_red1 = seg_img_red >= fuzzy_range_red[0]
        fuzzy_ids_red2 = seg_img_red <= fuzzy_range_red[1]
        fuzzy_ids_red = np.logical_and(fuzzy_ids_red1, fuzzy_ids_red2)

        fuzzy_ids_green = np.logical_and(seg_img_green >= fuzzy_range_green[0], seg_img_green <= fuzzy_range_green[1])

        fuzzy_ids_blue1 = (seg_img_blue >= fuzzy_range_blue[0])
        fuzzy_ids_blue2 = (seg_img_blue <= fuzzy_range_blue[1])
        fuzzy_ids_blue = np.logical_and(fuzzy_ids_blue1, fuzzy_ids_blue2)

        fuzzy_ids = np.logical_and(fuzzy_ids_red, fuzzy_ids_green)
        fuzzy_ids = np.logical_and(fuzzy_ids, fuzzy_ids_blue)

        # fuzzy_ids_red1_int = np.flatnonzero(fuzzy_ids_red1).size
        # fuzzy_ids_red2_int = np.flatnonzero(fuzzy_ids_red2).size
        # fuzzy_ids_red_int = np.flatnonzero(fuzzy_ids_red).size
        #
        # fuzzy_ids_green_int = np.flatnonzero(fuzzy_ids_green).size
        #
        # fuzzy_ids_blue1_int = np.flatnonzero(fuzzy_ids_blue1).size
        # fuzzy_ids_blue2_int = np.flatnonzero(fuzzy_ids_blue2).size
        # fuzzy_ids_blue_int = np.flatnonzero(fuzzy_ids_blue).size
        #
        # fuzzy_ids_int = np.flatnonzero(fuzzy_ids).size

        # print('actual_col: {}'.format(actual_col))
        #
        # print('fuzzy_range_red: {}'.format(fuzzy_range_red))
        # print('fuzzy_range_blue: {}'.format(fuzzy_range_blue))
        #
        # print('fuzzy_ids_red: {}, {}, {}'.format(fuzzy_ids_red1_int, fuzzy_ids_red2_int, fuzzy_ids_red_int))
        # print('fuzzy_ids_blue: {}, {}, {}'.format(fuzzy_ids_blue1_int, fuzzy_ids_blue2_int, fuzzy_ids_blue_int))

        seg_img_out[fuzzy_ids] = actual_col
        seg_img_raw[fuzzy_ids] = _id

        class_to_ids[_id] = fuzzy_ids

        # print()

    if check_equality:
        seg_img_rec = np.zeros_like(seg_img)
        for _class_id, _col in class_to_color.items():
            seg_img_rec[seg_img_raw == _class_id] = _col

        if not np.array_equal(seg_img, seg_img_rec):
            print("seg_img and seg_img_rec are not equal")

            cv2.imshow('seg_img', seg_img)
            cv2.imshow('seg_img_rec', seg_img_rec)

            cv2.waitKey(0)
            raise AssertionError("seg_img and seg_img_rec are not equal")

    return seg_img_out, seg_img_raw, class_to_ids


def convert_to_raw_mask(seg_img, n_classes, seg_src_file, class_to_color, class_to_ids):
    seg_img_flat = seg_img.reshape((-1, 3))
    seg_vals, seg_val_indxs = np.unique(seg_img_flat, return_index=1, axis=0)
    seg_vals = list(seg_vals)
    # seg_val_indxs = list(seg_val_indxs)

    seg_img_raw = np.zeros(seg_img.shape[:2], dtype=np.uint8)

    n_seg_vals = len(seg_vals)

    assert n_seg_vals == n_classes, \
        "number of classes is less than the number of unique pixel values in {}".format(seg_src_file)

    color_to_class = {
        v: k for k, v in class_to_color.items()
    }

    for seg_val_id, seg_val in enumerate(seg_vals):
        # print('{} --> {}'.format(seg_val, seg_val_id))
        class_id = color_to_class[tuple(seg_val)]
        _ids = class_to_ids[class_id]
        seg_img_raw[_ids] = class_id

    return seg_img_raw, seg_vals


def resize_ar(src_img, width=0, height=0, placement_type=0, bkg_col=None):
    """
    resize an image to the given size while maintaining its aspect ratio and adding a black border as needed;
    if either width or height is omitted or specified as 0, it will be automatically computed from the other one;
    both width and height cannot be omitted;

    :param src_img: image to be resized
    :type src_img: np.ndarray

    :param width: desired width of the resized image; will be computed from height if omitted
    :type width: int

    :param height: desired height of the resized image; will be computed from width if omitted
    :type height: int

    :param return_factors: return the multiplicative resizing factor along with the
    position of the source image within the returned image with borders
    :type height: int

    :param placement_type: specifies how the source image is to be placed within the returned image if
    borders need to be added to achieve the target size;
    0: source image is top-left justified;
    1: source image is center-middle justified;
    2: source image is bottom-right justified;
    :type placement_type: int

    :return: resized image with optional resizing / placement info
    :rtype: np.ndarray | tuple[np.ndarray, float, int, int]
    """
    src_height, src_width = src_img.shape[:2]

    try:
        n_channels = src_img.shape[2]
    except IndexError:
        n_channels = 1

    src_aspect_ratio = float(src_width) / float(src_height)

    assert width > 0 or height > 0, 'Both width and height cannot be zero'

    if height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type == 0:
            start_row = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type == 0:
            start_col = 0
        elif placement_type == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8).squeeze()

    if bkg_col is not None:
        dst_img.fill(bkg_col)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, ...] = src_img
    dst_img = cv2.resize(dst_img, (width, height))

    resize_factor = float(height) / float(dst_height)

    img_bbox = [start_col, start_row, start_col + src_width, start_row + src_height]
    return dst_img, resize_factor, img_bbox
