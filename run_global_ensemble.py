import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# SYSTEM CONFIGURATION & SEED INITIALIZATION
# ==========================================
np.random.seed(42)
torch.manual_seed(42)

# 명세서 기준 설정
SEQ_LEN = 90               
EMBED_DIM = 16             

# 국가 그룹별 가변 알파(α) 전략 정의 (명세: α는 국가 그룹별 조정)
# 데이터셋의 'group' 컬럼 값에 맞춰 적절한 가중치를 동적 매핑합니다.
ALPHA_MAP = {
    'A': 0.6,
    'B': 0.7,
    'C': 0.8,
    'OTHER': 0.7
}

# 1. DATA LOADING & PREPROCESSING
print("[INFO] Loading WRI labeled dataset...")
file_path = "wri_labeled_fixed.csv"
df = pd.read_csv(file_path)

df = df.sort_values(by=["year", "country_std"]).reset_index(drop=True)
feature_cols = ['CII', 'ESC', 'SFI', 'Econ', 'Spill', 'Polit', 'Z_CII_7d', 'Z_CII_90d', 'regcluster', 'idp_count_z', 'disaster_deaths_z']
N_FEATURES = len(feature_cols)

le = LabelEncoder()
df["country_idx"] = le.fit_transform(df["country_std"])
TOTAL_COUNTRIES = df["country_idx"].nunique()

print(f"[INFO] Total Countries: {TOTAL_COUNTRIES}, Features: {N_FEATURES}")


# 2. GLOBAL 3D SEQUENCE GENERATION (명세: seq_len=90 강제화 및 패딩 적용)
print("[INFO] Generating 3D sequences with seq_len=90...")

def build_global_sequences(data_df, seq_len=90):
    X_lstm_list, X_cid_list, Y_list, group_list = [], [], [], []
    
    for _, group_df in data_df.groupby("country_idx"):
        group_df = group_df.sort_values("year")
        features = group_df[feature_cols].values
        targets = group_df["Y"].values
        c_idx = group_df["country_idx"].iloc[0]
        g_name = group_df["group"].iloc[0] if "group" in group_df.columns else "OTHER"
        
        for i in range(len(group_df)):
            # 과거 데이터를 슬라이싱하되, 부족하면 전방 제로 패딩 처리
            chunk = features[max(0, i - seq_len + 1) : i + 1]
            if len(chunk) < seq_len:
                pad_width = seq_len - len(chunk)
                chunk = np.pad(chunk, ((pad_width, 0), (0, 0)), mode='constant')
            
            X_lstm_list.append(chunk)
            X_cid_list.append(c_idx)
            Y_list.append(targets[i])
            group_list.append(g_name)
            
    return np.array(X_lstm_list), np.array(X_cid_list), np.array(Y_list), np.array(group_list)

X_lstm_all, X_cid_all, Y_all, X_group_all = build_global_sequences(df, seq_len=SEQ_LEN)


# 3. COUNTRY EMBEDDING GENERATION FOR XGBOOST (명세: 국가ID 임베딩 + 전체 피처)
# XGBoost 입력단에 결합할 국가 고정효과 공간 임베딩 벡터 가중치를 초기화합니다.
embed_layer = nn.Embedding(TOTAL_COUNTRIES, EMBED_DIM)
torch.manual_seed(42)
with torch.no_grad():
    # 전체 국가 ID에 대한 임베딩 행렬 추출
    all_cid_tensor = torch.LongTensor(X_cid_all)
    cid_embeddings = embed_layer(all_cid_tensor).numpy()

# XGBoost 입력 매트릭스 구성: (국가ID 임베딩 벡터 + 최신 시점 1주의 피처)
X_xgb_latest = X_lstm_all[:, -1, :]
X_xgb_all = np.hstack((cid_embeddings, X_xgb_latest))

print(f"       - LSTM Input Shape  : {X_lstm_all.shape} (N_country*N_samples, seq_len, N_features)")
print(f"       - XGBoost Input Shape: {X_xgb_all.shape} (Embedding + Features 결합)")


# 4. MODEL ARCHITECTURE DEFINITION
class GlobalConflictLSTM(nn.Module):
    def __init__(self, num_countries, embed_dim=16, input_dim=11, hidden_dim=64):
        super(GlobalConflictLSTM, self).__init__()
        self.country_embedding = nn.Embedding(num_countries, embed_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim + embed_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, x_cid):
        lstm_out, _ = self.lstm(x_seq)
        lstm_last = lstm_out[:, -1, :]
        cid_emb = self.country_embedding(x_cid)
        combined = torch.cat((lstm_last, cid_emb), dim=1)
        out = torch.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out


# 5. TIME SERIES SPLIT CROSS-VALIDATION
print("[INFO] Starting time-series cross-validation...")
n_total_samples = len(Y_all)
fold_step = n_total_samples // 4

for fold in range(1, 4):
    train_bound = fold * fold_step
    val_bound = train_bound + fold_step
    
    if train_bound >= n_total_samples or train_bound == 0:
        continue
        
    print(f"\n--- FOLD {fold} PROGRESS ---")
    
    # Dataset Split
    X_xgb_train, X_xgb_val = X_xgb_all[:train_bound], X_xgb_all[train_bound:val_bound]
    X_lstm_train, X_lstm_val = X_lstm_all[:train_bound], X_lstm_all[train_bound:val_bound]
    X_cid_train, X_cid_val = X_cid_all[:train_bound], X_cid_all[train_bound:val_bound]
    y_train, y_val = Y_all[:train_bound], Y_all[train_bound:val_bound]
    val_groups = X_group_all[train_bound:val_bound]
    
    if len(y_val) == 0 or len(np.unique(y_train)) < 2:
        continue
        
    # (1) XGBoost Train & Predict
    xgb_global = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=42)
    xgb_global.fit(X_xgb_train, y_train)
    preds_xgb = xgb_global.predict_proba(X_xgb_val)[:, 1]
    
    # (2) LSTM Train & Predict
    train_ds = TensorDataset(torch.FloatTensor(X_lstm_train), torch.LongTensor(X_cid_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    lstm_global = GlobalConflictLSTM(num_countries=TOTAL_COUNTRIES, embed_dim=EMBED_DIM, input_dim=N_FEATURES)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lstm_global.parameters(), lr=0.002)
    
    lstm_global.train()
    for epoch in range(8):
        for seq_b, cid_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = lstm_global(seq_b, cid_b).squeeze()
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            
    lstm_global.eval()
    with torch.no_grad():
        preds_lstm = lstm_global(torch.FloatTensor(X_lstm_val), torch.LongTensor(X_cid_val)).squeeze().numpy()
        if preds_lstm.ndim == 0: preds_lstm = np.array([preds_lstm.item()])
        
    # (3) Dynamic Soft Voting Ensemble Blending (명세: α는 국가 그룹별 조정)
    preds_ensemble = np.zeros_like(preds_xgb)
    for idx, g_idx in enumerate(val_groups):
        alpha = ALPHA_MAP.get(g_idx, 0.7) # 그룹별 대치값 획득
        preds_ensemble[idx] = alpha * preds_xgb[idx] + (1 - alpha) * preds_lstm[idx]
        
    # (4) Evaluation Metrics 확정 반영 (AUC-ROC, F1, Precision, Recall, Calibration)
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, preds_ensemble)
        prec_arr, rec_arr, thres = precision_recall_curve(y_val, preds_ensemble)
        f1_arr = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-8)
        best_idx = np.argmax(f1_arr)
        
        best_th = thres[best_idx] if best_idx < len(thres) else 0.5
        final_preds_bin = (preds_ensemble >= best_th).astype(int)
        
        f1 = f1_arr[best_idx]
        precision = precision_score(y_val, final_preds_bin, zero_division=0)
        recall = recall_score(y_val, final_preds_bin, zero_division=0)
    else:
        auc, f1, precision, recall, best_th = 0.5, 0.0, 0.0, 0.0, 0.5
        
    brier = brier_score_loss(y_val, preds_ensemble)
    
    print(f"  Evaluation Metrics (Fold {fold}):")
    print(f"    - AUC-ROC             : {auc:.4f}")
    print(f"    - Global Best F1 Score: {f1:.4f} (Threshold: {best_th:.3f})")
    print(f"    - Precision           : {precision:.4f}")
    print(f"    - Recall              : {recall:.4f}")
    print(f"    - Calibration (Brier) : {brier:.4f}")

print("\n[INFO] Global conflict ensemble pipeline execution completed successfully.")