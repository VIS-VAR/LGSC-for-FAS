import paddle.fluid as fluid
from models.scan import SCAN
from utils.runner import Runnner
from dataset.faceforensics import FaceForensics

model_cfg = dict(
    backbone=dict(
        depth=18,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1),
    neck=dict(
        norm_cfg=dict(type='IN')),
    head=dict(
        depth=18,
        out_indices=(3,),
        norm_cfg=dict(type='BN'),
        dropout=0.5),
    train_cfg=dict(
        w_cls=5.0,
        w_tri=1.0,
        w_reg=5.0,
        with_mask=False),
    test_cfg=dict(
        thr=0.5),
    pretrained='./pretrained/resnet18-torch',
)

checkpoint_cfg = dict(
    work_dir='./work_dir/test/',
    load_from='./work_dir/ff_add_val/Best_model',
    save_interval=30000,
    eval_interval=200,
    log_interval=10,
    eval_type='acc'
)

optimizer_cfg = dict(
    lr=0.0005,
    type='Adam',
    warmup_iter=1000,
    decay_epoch=[5, 8, 12],
    decay=0.3,
    regularization=0.0005,
)

extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.8, 1.2),
        hue_delta=16),
    random_erasing=dict(
        probability=0.5,
        area=(0.01, 0.03),
        mean=(80, 80, 80)),
    random_cutout=dict(
        probability=0.5,
        max_edge=20),
    ramdom_rotate=dict(
        probability=0.5,
        angle=30),
    ramdom_crop=dict(
        probability=0.5,
        w_h=(0.12, 0.12))
)

data_root = 'Path/FaceForensics/data/'
test_dataset = FaceForensics(
    img_prefix=data_root,
    ann_file=data_root + 'faceforensics_benchmark.txt',
    #ann_file=data_root + 'c40_val.txt',
    img_scale=(224, 224),
    img_norm_cfg=dict(mean=(100, 100, 100), std=(80, 80, 80)),
    extra_aug=dict(),
    test_mode=True,
    crop_face=0.10,
)


with fluid.dygraph.guard():
    model = SCAN(**model_cfg)

runner = Runnner(
    model,
    test_dataset,
    batch_size=96,
    checkpoint_config=checkpoint_cfg,
    optimizer_config=optimizer_cfg)

runner.test(thr='mid')

