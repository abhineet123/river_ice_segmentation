import os, sys
from utils import processArguments

if len(sys.argv) < 3:
    raise SystemError('Names of the two folders to be merged must be provided')
src_dir = sys.argv[1]
dst_dir = sys.argv[2]

params = {
    'db_root_dir': '/data/617/images',
    'seq_name': 'Training',
    'fname_templ': 'img',
    'n_frames': 50,
    'start_id': 0,
    'end_id': -1,
    'rm_src': 1,
}

processArguments(sys.argv[3:], params)
db_root_dir = params['db_root_dir']
fname_templ = params['fname_templ']
n_frames = params['n_frames']
start_id = params['start_id']
end_id = params['end_id']
rm_src = params['rm_src']

if end_id < start_id:
    end_id = n_frames - 1

src_dir = os.path.join(db_root_dir, src_dir)
dst_dir = os.path.join(db_root_dir, dst_dir)

print('db_root_dir: ', db_root_dir)
print('src_dir: ', src_dir)
print('dst_dir: ', dst_dir)

dst_img_dir = os.path.join(dst_dir, 'images/')
dst_labels_dir = os.path.join(dst_dir, 'labels/')

if not os.path.isdir(dst_img_dir):
    print('Destination image folder does not exist. Creating it...')
    os.makedirs(dst_img_dir)

if not os.path.isdir(dst_labels_dir):
    print('Destination labels folder does not exist. Creating it...')
    os.makedirs(dst_labels_dir)

command = 'mv {:s}/{:s}_* {:s}/'.format(src_dir, fname_templ, dst_dir)
# print('command: ', command)

os.system(command)

for i in range(start_id, end_id + 1):
    src_img_list = os.path.join(src_dir, 'images', '{:s}_{:d}_*'.format(fname_templ, i + 1))


    # print('src_img_list: ', src_img_list)
    # print('dst_img_dir: ', dst_img_dir)

    os.system('mv {:s} {:s}'.format(src_img_list, dst_img_dir))

    src_labels_list = os.path.join(src_dir, 'labels', '{:s}_{:d}_*'.format(fname_templ, i + 1))

    os.system('mv {:s} {:s}'.format(src_labels_list, dst_labels_dir))

if rm_src:
    os.system('rm -rf {:s}'.format(src_dir))

