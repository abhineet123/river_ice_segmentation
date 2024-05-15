from datetime import datetime
import os
import paramparse

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk
    # import tkinter as Tk

class Params:

    def __init__(self):
        self.list_path = 'list'
        self.out_path = 'list_out'
        self.sep = '\t'


def main():
    _params = Params()
    paramparse.process(_params)

    in_files = []
    try:
        in_txt = Tk().clipboard_get()  # type: str
    except BaseException as e:
        print('Tk().clipboard_get() failed: {}'.format(e))
    else:
        in_files = [k.strip() for k in in_txt.split('\n') if k.strip()]

    if not in_files or not all(os.path.isfile(in_file) for in_file in in_files):
        in_files = open(_params.list_path, 'r').read().splitlines()

    all_in_lines = [open(in_file, 'r').read().splitlines() for in_file in in_files]

    out_lines = []
    for all_curr_lines in zip(*all_in_lines):
        out_line = _params.sep.join(all_curr_lines)

        out_lines.append(out_line)

    out_txt = '\n'.join(out_lines)

    open(_params.out_path, 'w').write(out_txt)

    # print('out_txt:\n{}'.format(out_txt))

    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
