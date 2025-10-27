# scripts/utils_config.py

import os, sys, json, pickle, random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import optuna
from optuna.trial import Trial

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- [0. g-mlp 모듈 경로] --------------------
# 사용자의 환경에 맞게 GMLP_DIR 경로를 수정하세요.
GMLP_DIR = "/home/minji/g-mlp"
if GMLP_DIR not in sys.path:
    sys.path.append(GMLP_DIR)
# gMLP 클래스는 model_core.py에서 import 합니다.

# -------------------- [1. 공통 유틸/환경] --------------------
def set_seed(seed=700):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- [2. 아티팩트/인덱스 저장/로드] --------------------
def save_split_indices(out_dir, tag, split_indices):
    os.makedirs(out_dir, exist_ok=True)
    train_idx, val_idx, test_idx = split_indices
    np.save(os.path.join(out_dir, f"train_idx_{tag}.npy"), np.array(train_idx, dtype=np.int64))
    np.save(os.path.join(out_dir, f"val_idx_{tag}.npy"), np.array(val_idx, dtype=np.int64))
    np.save(os.path.join(out_dir, f"test_idx_{tag}.npy"), np.array(test_idx, dtype=np.int64))

def load_split_indices(path, tag):
    train_idx = np.load(os.path.join(path, f"train_idx_{tag}.npy"))
    val_idx = np.load(os.path.join(path, f"val_idx_{tag}.npy"))
    test_idx = np.load(os.path.join(path, f"test_idx_{tag}.npy"))
    return train_idx, val_idx, test_idx

def save_config(cfg_path, config: dict):
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        return json.load(f)

def save_scaler(path, scaler):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
