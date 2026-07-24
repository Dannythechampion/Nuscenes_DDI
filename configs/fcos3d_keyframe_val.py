_base_ = (
    '/home/hansol/work/mmdetection3d/configs/fcos3d/'
    'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py'
)

data_root = '/home/hansol/data/nuscenes/'
subset_ann = 'nuscenes_keyframes_infos_val_subset.pkl'

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(data_root=data_root, ann_file=subset_ann))

test_evaluator = dict(
    _delete_=True,
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + subset_ann,
    metric='bbox',
    format_only=True,
    jsonfile_prefix='/home/hansol/results/fcos3d')

load_from = None
