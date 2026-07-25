_base_ = (
    '/home/hansol/work/mmdetection3d/configs/pgd/'
    'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d_finetune.py'
)

data_root = '/mnt/c/Users/UserK/Documents/Intern Project/data/nuscenes-val150/'
ann_file = 'nuscenes_camera_val150_infos_val_subset.pkl'

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(data_root=data_root, ann_file=ann_file))

test_evaluator = dict(
    _delete_=True,
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + ann_file,
    modality=dict(use_camera=True, use_lidar=False),
    metric='bbox',
    format_only=False,
    jsonfile_prefix='/home/hansol/results/val150/pgd')

load_from = None
