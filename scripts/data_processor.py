# scripts/data_processor.py

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm

# utils_config에서 필요한 모듈 import
from .utils_config import Chem, DataStructs, AllChem, MACCSkeys, rdMolDescriptors, Descriptors, MoleculeDescriptors

# -------------------- [2. 피처 생성 유틸] --------------------
def to_numpy_bitvect(bitvect, n_bits=None, drop_first=False):
    if n_bits is None:
        n_bits = bitvect.GetNumBits()
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    if drop_first:
        arr = arr[1:]
    return arr.astype(np.float32)

def get_ecfp(mol, radius=2, nbits=1024):
    return to_numpy_bitvect(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), n_bits=nbits)

def get_maccs(mol):
    bv = MACCSkeys.GenMACCSKeys(mol)
    return to_numpy_bitvect(bv, n_bits=bv.GetNumBits(), drop_first=True)

def get_avalon(mol, nbits=512):
    from rdkit.Avalon import pyAvalonTools
    return to_numpy_bitvect(pyAvalonTools.GetAvalonFP(mol, nbits), n_bits=nbits)

def get_topological_torsion(mol, nbits=1024):
    bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
    return to_numpy_bitvect(bv, n_bits=nbits)

def get_rdkit_desc(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([d[0] for d in Descriptors._descList])
    try:
        descs = calc.CalcDescriptors(mol)
        descs = np.array(descs, dtype=np.float32)
        descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        descs = np.zeros(len(Descriptors._descList), dtype=np.float32)
    return descs

def get_rdkit_descriptor_length():
    return len(Descriptors._descList)

# -------------------- [3. 임베딩 로드 & 차원 감지] --------------------
def load_molecular_embeddings(embed_paths: dict):
    embed_data, embed_dims = {}, {}
    
    for name, path in embed_paths.items(): 
        try:
            df = pd.read_csv(path)
        except Exception:
            embed_data[name] = {}
            embed_dims[name] = 0
            continue
        
        def canon(s):
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m, canonical=True) if m else None

        df['smiles'] = df['smiles'].apply(canon)
        df = df.dropna(subset=['smiles']).reset_index(drop=True)

        embed_cols = [c for c in df.columns if c != 'smiles']
        dim = len(embed_cols)
        
        embed_dims[name] = dim
        embed_data[name] = {
            row['smiles']: row[embed_cols].to_numpy(dtype=np.float32, copy=False)
            for _, row in df.iterrows()
        }
    
    return embed_data, embed_dims

# -------------------- [4. 기대 차원 계산 + 안전 결합] --------------------
def compute_expected_dims(fp_types, embed_dims: dict):
    expected = OrderedDict()
    for t in fp_types:
        if t == 'ecfp':
            expected[t] = 1024
        elif t == 'avalon':
            expected[t] = 512
        elif t == 'maccs':
            expected[t] = 166
        elif t == 'tt':
            expected[t] = 1024
        elif t == 'rdkit':
            expected[t] = get_rdkit_descriptor_length()
            
        elif 'mole' in t:
            default_dim = 768
            expected[t] = embed_dims.get(t, default_dim) if embed_dims.get(t, 0) > 0 else default_dim
            
        elif 'scage' in t:
            default_dim = 512
            expected[t] = embed_dims.get(t, default_dim) if embed_dims.get(t, 0) > 0 else default_dim
            
        else:
            raise ValueError(f"Unknown fp_type: {t}")
    return expected

def safe_fit_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec is None:
        return np.zeros(target_dim, dtype=np.float32)
    vec = vec.astype(np.float32, copy=False)
    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    cur = vec.shape[0]
    if cur == target_dim:
        return vec
    elif cur < target_dim:
        pad = np.zeros(target_dim - cur, dtype=np.float32)
        return np.concatenate([vec, pad], axis=0)
    else:
        return vec[:target_dim]

def make_feature_vector(mol, smiles, fp_types, expected_dims, embed_dicts):
    chunks = []
    for t in fp_types:
        dim = expected_dims[t]
        try:
            if t == 'ecfp':
                vec = get_ecfp(mol, radius=2, nbits=dim)
            elif t == 'avalon':
                vec = get_avalon(mol, nbits=dim)
            elif t == 'maccs':
                vec = get_maccs(mol)
            elif t == 'tt':
                vec = get_topological_torsion(mol, nbits=dim)
            elif t == 'rdkit':
                vec = get_rdkit_desc(mol)
            elif 'scage' in t or 'mole' in t:
                vec = embed_dicts.get(t, {}).get(smiles, None)
            else:
                vec = None
        except Exception:
            vec = None
        chunks.append(safe_fit_to_dim(vec, dim))
    feat = np.concatenate(chunks, axis=0)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat

# -------------------- [5. Dataset] --------------------
class ScageConcatDataset(data.Dataset):
    def __init__(self, label_path, embed_paths: dict, fp_types, expected_dims=None):
        df = pd.read_csv(label_path)
        df = df[['smiles', 'p_np']].rename(columns={'smiles': 'smiles', 'p_np': 'label'})
        df['label'] = df['label'].replace({'BBB-': 0, 'BBB+': 1})
        df = df.drop_duplicates(subset='smiles').reset_index(drop=True)

        self.embed_dicts, embed_dims = load_molecular_embeddings(embed_paths)

        if expected_dims is None:
            expected_dims = compute_expected_dims(fp_types, embed_dims) 
        self.expected_dims = expected_dims
        self.fp_types = list(fp_types)

        def canon(s):
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m, canonical=True) if m else None

        df['smiles'] = df['smiles'].apply(canon)
        df = df.dropna(subset=['smiles']).reset_index(drop=True)

        features, labels, failed = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Features"):
            smi = row['smiles']
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed.append(smi)
                continue
            feat = make_feature_vector(mol, smi, self.fp_types, self.expected_dims, self.embed_dicts) 
            if feat is None or feat.ndim != 1:
                failed.append(smi)
                continue
            features.append(feat)
            labels.append(row['label'])

        self.features = torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.df = df[~df['smiles'].isin(failed)].reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
