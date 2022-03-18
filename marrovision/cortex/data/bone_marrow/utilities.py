import os
import torch
import torch.utils.data.dataloader
from .dataset import BoneMarrowDataset
from pathlib import Path
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import marrovision.cortex.data.bone_marrow.transformations
from .sampler import BoneMarrowBalancedSampler


def get_image_paths_per_label(dataset_root, label):
    filepaths = []
    for path in Path(os.path.join(dataset_root, label)).rglob('*.jpg'):
        if path.is_file():
            filepaths.append(path)
    return filepaths


def f1(x, y):
    return (2*x*y)/(x+y)


def get_results_comparison_table(stats):
    meta_df = pandas.DataFrame([
        dict(
            class_name="Band Neutrophils",
            class_abbreviation="NGB",
            baseline_precision_strict=54,
            baseline_recall_strict=65,
            baseline_support=9968,
        ),
        dict(
            class_name="Segmented neutrophils",
            class_abbreviation="NGS",
            baseline_precision_strict=92,
            baseline_recall_strict=71,
            baseline_support=29424,
        ),
        dict(
            class_name="Lymphocytes",
            class_abbreviation="LYT",
            baseline_precision_strict=90,
            baseline_recall_strict=70,
            baseline_support=26242,
        ),
        dict(
            class_name="Monocytes",
            class_abbreviation="MON",
            baseline_precision_strict=57,
            baseline_recall_strict=70,
            baseline_support=4040,
        ),
        dict(
            class_name="Eosinophils",
            class_abbreviation="EOS",
            baseline_precision_strict=85,
            baseline_recall_strict=91,
            baseline_support=5883,
        ),
        dict(
            class_name="Basophils",
            class_abbreviation="BAS",
            baseline_precision_strict=14,
            baseline_recall_strict=64,
            baseline_support=441,
        ),
        dict(
            class_name="Metamyelocytes",
            class_abbreviation="MMZ",
            baseline_precision_strict=30,
            baseline_recall_strict=64,
            baseline_support=3055,
        ),
        dict(
            class_name="Myelocytes",
            class_abbreviation="MYB",
            baseline_precision_strict=52,
            baseline_recall_strict=59,
            baseline_support=6557,
        ),
        dict(
            class_name="Promyelocytes",
            class_abbreviation="PMO",
            baseline_precision_strict=76,
            baseline_recall_strict=72,
            baseline_support=11994,
        ),
        dict(
            class_name="Blasts",
            class_abbreviation="BLA",
            baseline_precision_strict=75,
            baseline_recall_strict=65,
            baseline_support=11973,
        ),
        dict(
            class_name="Plasma cells",
            class_abbreviation="PLM",
            baseline_precision_strict=81,
            baseline_recall_strict=84,
            baseline_support=7629,
        ),
        dict(
            class_name="Smudge cells",
            class_abbreviation="KSC",
            baseline_precision_strict=28,
            baseline_recall_strict=90,
            baseline_support=42,
        ),
        dict(
            class_name="Other cells",
            class_abbreviation="OTH",
            baseline_precision_strict=22,
            baseline_recall_strict=84,
            baseline_support=294,
        ),
        dict(
            class_name="Artefacts",
            class_abbreviation="ART",
            baseline_precision_strict=82,
            baseline_recall_strict=74,
            baseline_support=19630,
        ),
        dict(
            class_name="Not identifiable",
            class_abbreviation="NIF",
            baseline_precision_strict=27,
            baseline_recall_strict=63,
            baseline_support=3538,
        ),
        dict(
            class_name="Proerythroblasts",
            class_abbreviation="PEB",
            baseline_precision_strict=57,
            baseline_recall_strict=63,
            baseline_support=2740,
        ),
        dict(
            class_name="Erythroblasts",
            class_abbreviation="EBO",
            baseline_precision_strict=88,
            baseline_recall_strict=82,
            baseline_support=27395,
        ),
        dict(
            class_name="Hairy cells",
            class_abbreviation="HAC",
            baseline_precision_strict=35,
            baseline_recall_strict=80,
            baseline_support=409,
        ),
        dict(
            class_name="Abnormal eosinophils",
            class_abbreviation="ABE",
            baseline_precision_strict=2,
            baseline_recall_strict=20,
            baseline_support=8,
        ),
        dict(
            class_name="Immature lymphocytes",
            class_abbreviation="LYI",
            baseline_precision_strict=8,
            baseline_recall_strict=53,
            baseline_support=65,
        ),
        dict(
            class_name="Faggot cells",
            class_abbreviation="FGC",
            baseline_precision_strict=17,
            baseline_recall_strict=63,
            baseline_support=47,
        ),
    ])
    meta_df['baseline_f1_strict'] = meta_df.apply(lambda x: f1(x['baseline_precision_strict'], x['baseline_recall_strict']), axis=1)

    for c in [f'baseline_{x}' for x in ['precision_strict', 'recall_strict', 'f1_strict']]:
        meta_df[c] = meta_df[c].apply(lambda x: x * 0.01)

    for c in ['model_precision', 'model_recall', 'model_f1', 'train_support']:
        meta_df[c] = meta_df.apply(lambda x: get_category_info(x['class_abbreviation'], stats)[c], axis=1)

    meta_df['model_is_better'] = meta_df.apply(lambda x: x['model_f1'] - x['baseline_f1_strict'], axis=1)

    return meta_df


def get_category_info(class_abbreviation, stats):
    class_index = stats[-1]['label_layout']['labels'].index(class_abbreviation)
    return {
        'model_precision': stats[-1]['prf'][None][0][class_index],
        'model_recall': stats[-1]['prf'][None][1][class_index],
        'model_f1': stats[-1]['prf'][None][2][class_index],
        'train_support': stats[-1]['prf'][None][3][class_index],}