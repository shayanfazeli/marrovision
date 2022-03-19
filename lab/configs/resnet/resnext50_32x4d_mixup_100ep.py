__code_root = '/home/shayan/phoenix/marrovision/'
__warehouse_root = '/home/shayan/warehouse/marrovision/'
__logdir = __warehouse_root + 'bone_marrow_cell_classification/resnext50_32x4d_mixup_100ep'
__data_dir = '/data/marrovision/BM_cytomorphology_data'


data = dict(
    interface='bone_marrow_cell_classification',
    args=dict(
        data_dir=__data_dir,
        test_ratio=0.2,
        batch_size=64,
        num_workers=10,
        train_transformation='train_transform_1',
        balanced_sample_count_per_category=10000,
        mixup_alpha=0.1,
        cutmix_alpha=0.0,
        kfold_config=dict(
            n_splits=5
        )
    )
)

model = dict(
    type='ClassifierWithTorchvisionBackbone',
    config=dict(
        backbone=dict(
            type='resnext50_32x4d',
            args=dict(pretrained=False)
        ),
        head=dict(
            modules=[
                dict(
                    type='Linear',
                    args=dict(
                        in_features=2048,
                        out_features=21,
                        bias=True
                    )
                )
            ]
        ),
        loss=[
            dict(
                name='ce_loss',
                type='CrossEntropyLoss',
                target_variable='gt_score',
                weight=1.0,
                args=dict()
            )
        ],
        eval_loss=[
            dict(
                name='ce_loss',
                type='CrossEntropyLoss',
                target_variable='label_index',
                weight=1.0,
                args=dict()
            )
        ]
    ),
)

trainer = dict(
    type='StandardSemiSupervisedClassificationTrainer',
    config=dict(
        optimizer=dict(
            type="SGD",
            args=dict(
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-6
            )
        ),
        epoch_scheduler=dict(
            type="CosineAnnealingLR",
            args=dict(
                T_max=100,
                eta_min=0,
                last_epoch=-1
            )
        ),
        max_epochs=100,
        checkpointing=dict(
            checkpointing_interval=5,
            repo=__logdir,
        ),
    ),
)
