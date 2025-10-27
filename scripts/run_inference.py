# run_inference.py

import os
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict

# 분리된 모듈 import
from scripts.utils_config import set_seed, device, load_config, load_scaler
from scripts.data_processor import ScageConcatDataset
from scripts.model_core import MultiModalGMLPFromFlat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# -------------------- [A. 앙상블 추론 및 평가 설정] --------------------
set_seed(700)
seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
split_mode = "scaffold"
base_dir = "../artifacts"

# 새 데이터 경로 설정 (run_train.py와 경로가 다를 수 있음)
NEW_LABEL_PATH = "/home/minji/scage/BBB/data/my_label.csv"
NEW_EMBED_PATHS = {
    'scage1': '/home/minji/scage/BBB/data/my_embed.csv',
    'mole': '/home/minji/mole_public/MolE_embed_base_my.csv'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- [1. 공통 특징 및 설정 로드] --------------------
# 특징 타입과 차원 정보를 얻기 위해 시드 42의 설정 로드 (모델 구조 파악용)
ARTIFACT_TAG_42 = f"final_{split_mode}_seed42" # run_train.py에서 사용한 tag 형식
CONFIG_42_PATH = f"{base_dir}/feature_config_{ARTIFACT_TAG_42}.json"

try:
    config_42 = load_config(CONFIG_42_PATH)
    mod_dims = OrderedDict(config_42['mod_dims'])
    fp_types = config_42['fp_types']
    rd_slice_info = config_42['rd_slice']
except FileNotFoundError:
    print(f"🚨 에러: 초기 설정 파일 ({CONFIG_42_PATH})이 누락되었습니다. run_train.py를 먼저 실행하세요.")
    sys.exit(1)

# 새 데이터셋 특징 생성 (Raw Features)
new_dataset_raw = ScageConcatDataset(NEW_LABEL_PATH, NEW_EMBED_PATHS, fp_types, expected_dims=mod_dims)
raw_features = new_dataset_raw.features.clone()
raw_df = new_dataset_raw.df[['smiles', 'label']].copy()
dfs = []


# -------------------- [2. 시드별 예측 수행 및 저장] --------------------
print("="*80)
print("--- Starting Soft Voting Inference ---")
print(f"New Data Samples: {len(raw_df)}")
print("="*80)

for seed in seeds:
    artifact_tag = f"optimized_{split_mode}_seed{seed}"
    
    # 파일 경로 정의
    cfg_path    = f"{base_dir}/feature_config_{artifact_tag}.json"
    model_path  = f"{base_dir}/gmlp_best_model_{artifact_tag}.pth"
    scaler_path = f"{base_dir}/rdkit_scaler_{artifact_tag}.pkl"

    try:
        # A. 설정 및 스케일러 로드
        cfg = load_config(cfg_path)
        hparams = cfg['model_hparams']
        scaler = load_scaler(scaler_path) if os.path.exists(scaler_path) else None
        
        # B. 모델 초기화 및 가중치 로드
        model = MultiModalGMLPFromFlat(
            mod_dims=mod_dims, **hparams
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # C. 데이터 복제 및 정규화 적용 (시드별 Scaler 적용)
        X_pred = raw_features.clone()
        if rd_slice_info is not None and scaler is not None:
            rd_start, rd_end = rd_slice_info
            X_pred[:, rd_start:rd_end] = torch.tensor(
                scaler.transform(X_pred[:, rd_start:rd_end].cpu().numpy()), 
                dtype=torch.float32
            )
            
        # D. 예측 수행
        with torch.no_grad():
            X_tensor = X_pred.to(DEVICE)
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # E. 결과 정리 및 리스트에 추가
        df_result = raw_df[['smiles', 'label']].copy()
        df_result[f'prob_{seed}'] = probs
        dfs.append(df_result)
        print(f"✅ Seed {seed} model prediction successful.")
        
    except FileNotFoundError:
        print(f"⚠️ 경고: Seed {seed}의 아티팩트 파일이 누락되어 건너뜁니다.")
    except Exception as e:
        print(f"❌ 에러: Seed {seed} 모델 로드/예측 중 오류 발생: {e}")


# -------------------- [3. Soft Voting 앙상블 및 평가] --------------------
if len(dfs) < 2:
    print("❌ 앙상블을 수행하기에 충분한 모델이 로드되지 않았습니다. (최소 2개 모델 필요)")
else:
    # A. smiles/label 기준으로 병합
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=['smiles', 'label'], how='inner')

    # B. Soft Voting (확률 평균)
    prob_cols = [col for col in merged.columns if col.startswith('prob_')]
    merged['ensemble_prob'] = merged[prob_cols].mean(axis=1)
    merged['ensemble_pred'] = (merged['ensemble_prob'] > 0.5).astype(int)

    # C. 성능 평가
    y_true = merged['label']
    y_pred = merged['ensemble_pred']
    y_prob = merged['ensemble_prob']

    # D. 결과 출력
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Specificity 계산
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    print("\n" + "="*80)
    print("--- Soft Voting Ensemble Performance ---")
    print("="*80)
    print(f"Models used: {len(prob_cols)}")
    print(f"Accuracy:        {acc:.3f}")
    print(f"F1-score:        {f1:.3f}")
    print(f"ROC-AUC:         {auc:.3f}")
    print(f"MCC:             {mcc:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # E. Confusion Matrix 시각화
    import seaborn as sns # seaborn import
    import matplotlib.pyplot as plt # plt import
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0 (BBB-)', 'Pred 1 (BBB+)'],
                yticklabels=['True 0 (BBB-)', 'True 1 (BBB+)'])
    plt.title("Confusion Matrix - Soft Voting Ensemble")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # F. 결과 저장
    save_path = f"{base_dir}/optimized_bbb_ensemble_result.csv"
    merged.to_csv(save_path, index=False)
    print(f"\n✅ Soft voting ensemble 결과 저장 완료 → {save_path}")
