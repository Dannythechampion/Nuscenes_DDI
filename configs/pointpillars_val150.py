_base_ = (
    '/home/hansol/work/mmdetection3d/configs/pointpillars/'
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py'
)

data_root = '/mnt/c/Users/UserK/Documents/Intern Project/data/nuscenes-val150/'
ann_file = 'nuscenes_lidar10_val150_infos_val_subset.pkl'

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
    modality=dict(use_camera=False, use_lidar=True),
    metric='bbox',
    format_only=False,
    jsonfile_prefix='/home/hansol/results/val150/pointpillars')

load_from = None
