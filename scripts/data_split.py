# scripts/data_split.py

import numpy as np
import torch
import torch.utils.data as data
import random

# utils_config, data_processor에서 필요한 요소 import
from .utils_config import Chem, MurckoScaffold, StandardScaler, set_seed
from .data_processor import ScageConcatDataset 

# -------------------- [6. 스플릿 + RDKit 정규화 (토글)] --------------------
def split_then_normalize(
    dataset: ScageConcatDataset,
    split_mode: str = "scaffold",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 700
):
    set_seed(seed)
    df = dataset.df.copy()

    def get_scaffold(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None

    df['scaffold'] = df['smiles'].apply(get_scaffold)
    groups = list(df.groupby('scaffold').groups.values())

    if split_mode == "scaffold":
        groups = sorted(groups, key=lambda g: len(g), reverse=True)
    elif split_mode == "random_scaffold":
        rnd = random.Random(seed)
        rnd.shuffle(groups)
    else:
        raise ValueError("split_mode must be 'scaffold' or 'random_scaffold'")

    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio   * n))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        g = list(g)
        if len(train_idx) + len(g) <= train_cap:
            train_idx += g
        elif len(val_idx) + len(g) <= val_cap:
            val_idx += g
        else:
            test_idx += g

    def pick(idxs):
        return dataset.features[idxs], dataset.labels[idxs]

    X_train, y_train = pick(train_idx)
    X_val,   y_val   = pick(val_idx)
    X_test,  y_test  = pick(test_idx)

    rd_start, rd_end = None, None
    offset = 0
    for t in dataset.fp_types:
        dim = dataset.expected_dims[t]
        if t == 'rdkit':
            rd_start, rd_end = offset, offset + dim
            break
        offset += dim

    scaler = None
    if rd_start is not None:
        scaler = StandardScaler().fit(X_train[:, rd_start:rd_end])
        X_train[:, rd_start:rd_end] = torch.tensor(scaler.transform(X_train[:, rd_start:rd_end]), dtype=torch.float32)
        X_val[:, rd_start:rd_end]   = torch.tensor(scaler.transform(X_val[:, rd_start:rd_end]),   dtype=torch.float32)
        X_test[:, rd_start:rd_end]  = torch.tensor(scaler.transform(X_test[:, rd_start:rd_end]),  dtype=torch.float32)

    return (
        data.TensorDataset(X_train, y_train),
        data.TensorDataset(X_val,   y_val),
        data.TensorDataset(X_test,  y_test),
        scaler, (rd_start, rd_end),
        (train_idx, val_idx, test_idx)
    )
