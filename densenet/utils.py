import os, time, sys
import numpy as np
import cv2

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


def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg = args[arg_id].split('=')
        if len(arg) != 2 or not arg[0] in params.keys():
            print('Invalid argument provided: {:s}'.format(args[arg_id]))
            return

        if not arg[1] or not arg[0] or arg[1] == '#':
            continue

        if arg[0].startswith('--'):
            arg[0] = arg[0][2:]

        if isinstance(params[arg[0]], (list, tuple)):
            # if not ',' in arg[1]:
            #     print('Invalid argument provided for list: {:s}'.format(arg[1]))
            #     return

            if arg[1] and ',' not in arg[1]:
                arg[1] = '{},'.format(arg[1])

            arg_vals = arg[1].split(',')
            arg_vals_parsed = []
            for _val in arg_vals:
                try:
                    _val_parsed = int(_val)
                except ValueError:
                    try:
                        _val_parsed = float(_val)
                    except ValueError:
                        _val_parsed = _val if _val else None

                if _val_parsed is not None:
                    arg_vals_parsed.append(_val_parsed)
            params[arg[0]] = arg_vals_parsed
        else:
            params[arg[0]] = type(params[arg[0]])(arg[1])


def resize_ar_old(src_img, width, height, return_factors=False):
    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
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
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
    dst_img = cv2.resize(dst_img, (width, height))
    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


class LogWriter:
    def __init__(self, fname):
        self.fname = fname

    def _print(self, _str):
        print(_str + '\n')
        with open(self.fname, 'a') as fid:
            fid.write(_str + '\n')


def print_and_write(_str, fname=None):
    sys.stdout.write(_str + '\n')
    sys.stdout.flush()
    if fname is not None:
        open(fname, 'a').write(_str + '\n')



def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def sort_key(fname):
    fname = os.path.splitext(fname)[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)
    nums = [int(s) for s in fname.split('_') if s.isdigit()]
    non_nums = [s for s in fname.split('_') if not s.isdigit()]
    key = ''
    for non_num in non_nums:
        if not key:
            key = non_num
        else:
            key = '{}_{}'.format(key, non_num)
    for num in nums:
        if not key:
            key = '{:08d}'.format(num)
        else:
            key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('key: ', key)
    return key


def put_text_with_background(img, text, fmt=None):
    font_types = {
        0: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1: cv2.FONT_HERSHEY_COMPLEX,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_PLAIN,
        4: cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        5: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        6: cv2.FONT_HERSHEY_SIMPLEX,
        7: cv2.FONT_HERSHEY_TRIPLEX,
        8: cv2.FONT_ITALIC,
    }
    loc = (5, 15)
    size = 1
    thickness = 1
    col = (255, 255, 255)
    bgr_col = (0, 0, 0)
    font_id = 0

    if fmt is not None:
        try:
            font_id = fmt[0]
            loc = tuple(fmt[1:3])
            size, thickness = fmt[3:5]
            col = tuple(fmt[5:8])
            bgr_col = tuple(fmt[8:])
        except IndexError:
            pass

    disable_bkg = any([k < 0 for k in bgr_col])

    # print('font_id: {}'.format(font_id))
    # print('loc: {}'.format(loc))
    # print('size: {}'.format(size))
    # print('thickness: {}'.format(thickness))
    # print('col: {}'.format(col))
    # print('bgr_col: {}'.format(bgr_col))
    # print('disable_bkg: {}'.format(disable_bkg))

    font = font_types[font_id]

    text_offset_x, text_offset_y = loc
    if not disable_bkg:
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=size, thickness=thickness)[0]
        box_coords = ((text_offset_x, text_offset_y + 5), (text_offset_x + text_width, text_offset_y - text_height))
        cv2.rectangle(img, box_coords[0], box_coords[1], bgr_col, cv2.FILLED)
    cv2.putText(img, text, loc, font, size, col, thickness)

    # cv2.imshow('putTextWithBackground', img)


def resize_ar(src_img, width=0, height=0, return_factors=False, bkg_col=0):
    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if height == 0 and width == 0:
        raise IOError('Both height and width cannot be zero')
    elif height == 0:
        height = int(width / src_aspect_ratio)
    elif width == 0:
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
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        start_row = 0

    dst_img = np.full((dst_height, dst_width, n_channels), bkg_col, dtype=np.uint8)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
    dst_img = cv2.resize(dst_img, (width, height))
    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


def read_data(images_path='', images_ext='', labels_path='', labels_ext='',
              images_type='source', labels_type='labels'):
    src_file_list = src_labels_list = None
    total_frames = 0

    if images_path and images_ext:
        print('Reading {} images from: {}'.format(images_type, images_path))
        src_file_list = [os.path.join(images_path, k) for k in os.listdir(images_path) if
                         k.endswith('.{:s}'.format(images_ext))]
        total_frames = len(src_file_list)

        assert total_frames > 0, 'No input frames found'

        print('total_frames: {}'.format(total_frames))
        src_file_list.sort(key=sort_key)

    if labels_path and labels_ext:
        print('Reading {} images from: {}'.format(labels_type, labels_path))
        src_labels_list = [os.path.join(labels_path, k) for k in os.listdir(labels_path) if
                           k.endswith('.{:s}'.format(labels_ext))]
        if src_file_list is not None:
            assert total_frames == len(src_labels_list), 'Mismatch between no. of labels and images'
        else:
            total_frames = len(src_labels_list)

        src_labels_list.sort(key=sort_key)

    return src_file_list, src_labels_list, total_frames


def getDateTime():
    return time.strftime("%y%m%d_%H%M", time.localtime())
