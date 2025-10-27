# run_train.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from collections import OrderedDict

# 분리된 모듈 import
from scripts.utils_config import set_seed, device, save_split_indices, save_config, save_scaler
from scripts.data_processor import ScageConcatDataset
from scripts.data_split import split_then_normalize
from scripts.model_core import MultiModalGMLPFromFlat, train_model, eval_model

# -------------------- [A. 클래스 분포 시각화 유틸] --------------------
def plot_class_distribution(train_labels, val_labels, test_labels, split_mode, seed):
    import seaborn as sns # seaborn import
    
    labels_map = {0: 'Negative', 1: 'Positive'}
    train_counts = pd.Series(train_labels).map(labels_map).value_counts(normalize=True).sort_index()
    val_counts = pd.Series(val_labels).map(labels_map).value_counts(normalize=True).sort_index()
    test_counts = pd.Series(test_labels).map(labels_map).value_counts(normalize=True).sort_index()

    counts_df = pd.DataFrame({
        'Train': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    }).fillna(0)

    os.makedirs("./artifacts", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    counts_df.T.plot(kind='bar', stacked=False, ax=ax, rot=0)
    ax.set_title(f'Class Distribution by Split ({split_mode}, Seed: {seed})')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Dataset Split')
    ax.legend(title='Class')
    plt.tight_layout()
    plt.savefig(f'./artifacts/class_distribution_{split_mode}_seed{seed}.png')
    plt.close()


if __name__ == "__main__":
    set_seed(700) # 초기 설정 시드 고정
    
    # (A) 실험 스위치
    split_mode = "scaffold" # "scaffold" | "random_scaffold"
    
    # (A-1) 반복 실험을 위한 시드 리스트
    seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    
    # (A-2) 결과를 저장할 리스트
    results = []

    # (L) 반복문 시작
    for seed in seeds:
        print("\n" + "="*80)
        print(f"--- [STARTING NEW RUN] Split Mode: {split_mode}, Seed: {seed} ---")
        print("="*80)
        
        set_seed(seed) # 시드 고정
        
        # (B) 경로/설정 — BBBP(2039)만 사용
        label_path = '/home/minji/scage/BBB/data/bench_label.csv'
        mole_path = '/home/minji/mole_public/MolE_embed_base_bbb.csv'
        
        embed_paths = {
            'scage1': '/home/minji/scage/BBB/data/bench_embed.csv',
            'scage2' : '/home/minji/scage/BBB/data/bench_atom_embed.csv',
            'mole' : mole_path
        }
        fp_types = ['rdkit', 'scage1', 'mole']

        # (C) Dataset & dims
        dataset = ScageConcatDataset(label_path, embed_paths, fp_types=fp_types)
        expected_dims = dataset.expected_dims
        mod_dims = OrderedDict((t, expected_dims[t]) for t in fp_types)

        # (D) Split + RDKit normalize (8:1:1)
        train_ds, val_ds, test_ds, scaler, (rd_start, rd_end), split_indices = split_then_normalize(
            dataset, split_mode=split_mode, train_ratio=0.8, val_ratio=0.1, seed=seed
        )

        # (E) 스플릿 저장
        split_dir = "./splits"
        tag = f"{split_mode}_seed{seed}"
        save_split_indices(split_dir, tag, split_indices)
        print(f"[Save] split indices -> {split_dir} (tag={tag})")

        # (F) DataLoader
        train_loader = data.DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader   = data.DataLoader(val_ds,   batch_size=128, shuffle=False)
        test_loader  = data.DataLoader(test_ds,  batch_size=128, shuffle=False)
        
        # --- 클래스 분포 분석 ---
        y_train = train_ds.tensors[1].cpu().numpy()
        y_val = val_ds.tensors[1].cpu().numpy()
        y_test = test_ds.tensors[1].cpu().numpy()
        
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()

        print(f"\n--- [Info] Class distribution for seed {seed} ---")
        print(f"Train: Positive={np.mean(y_train):.2f}, Negative={1-np.mean(y_train):.2f}")
        plot_class_distribution(y_train, y_val, y_test, split_mode, seed)

        # (G) Model
        set_seed(seed)
        model_hparams = {"d_model": 512, "d_ffn": 1048, "depth": 4, "dropout": 0.2, "use_gated_pool": True}
        model = MultiModalGMLPFromFlat(
            mod_dims=mod_dims, **model_hparams
        ).to(device)

        # (H) pos_weight 및 Optimizer/Loss 설정
        pos_weight = None
        try:
            if n_pos > 0:
                pos_weight = torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device)
                print(f"[Info] Using pos_weight={pos_weight.item():.4f} (neg/pos={n_neg}/{n_pos})")
        except Exception as e:
            print(f"[Warn] pos_weight auto-calc skipped: {e}")

        train_params = {"lr": 1e-4, "weight_decay": 1e-5, "num_epochs": 50, "patience": 10}
        optimizer = optim.Adam(model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

        # (I) Train
        print(f"--- [STEP 1] Training gMLP on BBBP (split_mode={split_mode}) ---")
        model = train_model(model, optimizer, train_loader, val_loader, loss_fn, 
                            num_epochs=train_params['num_epochs'], patience=train_params['patience'])
        
        # (J) Test eval
        print("\n--- [STEP 2] Evaluating on BBBP Test Split ---")
        metrics = eval_model(model, test_loader)
        
        # (J-1) 각 시드별 결과 저장
        metrics['seed'] = seed
        metrics['train_pos_ratio'] = np.mean(y_train)
        metrics['val_pos_ratio'] = np.mean(y_val)
        metrics['test_pos_ratio'] = np.mean(y_test)
        results.append(metrics)
        
        # (K) 아티팩트 저장
        model_path  = f"./artifacts/gmlp_best_model_{tag}.pth"
        cfg_path    = f"./artifacts/feature_config_{tag}.json"
        scaler_path = f"./artifacts/rdkit_scaler_{tag}.pkl"
        
        torch.save(model.state_dict(), model_path)
        cfg = {
            "fp_types": fp_types,
            "mod_dims": {k: int(v) for k, v in mod_dims.items()},
            "rd_slice": [rd_start, rd_end] if rd_start is not None else None,
            "split_mode": split_mode,
            "seed": seed,
            "model_hparams": model_hparams,
            "train_params": train_params,
            "class_balance": {"n_pos": int(n_pos), "n_neg": int(n_neg), 
                              "pos_weight": float(pos_weight.item()) if pos_weight is not None else None}
        }
        save_config(cfg_path, cfg)
        if scaler is not None:
            save_scaler(scaler_path, scaler)
    
    # (M) 최종 결과 요약 및 출력
    print("\n" + "="*80)
    print("--- [FINAL SUMMARY] Average Performance Across All Seeds ---")
    print("="*80)
    
    results_df = pd.DataFrame(results).set_index('seed')
    summary = results_df.mean(numeric_only=True)
    std_dev = results_df.std(numeric_only=True)
    
    for metric in summary.index:
        print(f"{metric:<18}| Mean: {summary[metric]:.4f} | Std Dev: {std_dev[metric]:.4f}")
    
    results_df.to_csv(f"./artifacts/multi_seed_results_{split_mode}.csv")
    print(f"\n[Save] Detailed results saved to ./artifacts/multi_seed_results_{split_mode}.csv")
    print("="*80)
