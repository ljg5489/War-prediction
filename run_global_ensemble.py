import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

# ==========================================
# SYSTEM CONFIGURATION & SEED INITIALIZATION
# ==========================================
np.random.seed(42)
torch.manual_seed(42)

SEQ_LEN = 1                
ALPHA = 0.8        

# 1. DATA LOADING & PREPROCESSING
print("[INFO] Loading WRI labeled dataset...")

file_path = "wri_labeled_fixed.csv"
df = pd.read_csv(file_path)

# Sort by year and country to prevent data leakage
df = df.sort_values(by=["year", "country_std"]).reset_index(drop=True)

# Define feature columns based on feature specification v3.1
feature_cols = ['CII', 'ESC', 'SFI', 'Econ', 'Spill', 'Polit', 'Z_CII_7d', 'Z_CII_90d', 'regcluster', 'idp_count_z', 'disaster_deaths_z']
N_FEATURES = len(feature_cols)

# Encode country identifiers into sequential integers for the embedding layer
le = LabelEncoder()
df["country_idx"] = le.fit_transform(df["country_std"])
TOTAL_COUNTRIES = df["country_idx"].nunique()

print(f"[INFO] Dataset loaded successfully.")
print(f"       - Total Countries: {TOTAL_COUNTRIES}")
print(f"       - Input Features: {N_FEATURES}")
print(f"       - Total Instances: {df.shape[0]}")


# 2. GLOBAL 3D SEQUENCE GENERATION
print("[INFO] Transforming panel data into 3D sequences...")

def build_real_sequences(data_df, seq_len=1):
    X_lstm_list, X_cid_list, Y_list = [], [], []
    
    for c_idx, group in data_df.groupby("country_idx"):
        group = group.sort_values("year")
        
        features = group[feature_cols].values
        targets = group["Y"].values
        
        current_seq_len = seq_len
        if len(group) <= current_seq_len:
            current_seq_len = len(group) - 1
            
        if current_seq_len < 1:
            continue
            
        for i in range(len(group) - current_seq_len):
            chunk = features[i : i + current_seq_len]
            if len(chunk) < seq_len:
                pad_width = seq_len - len(chunk)
                chunk = np.pad(chunk, ((pad_width, 0), (0, 0)), mode='constant')
                
            X_lstm_list.append(chunk)
            X_cid_list.append(c_idx)
            Y_list.append(targets[i + current_seq_len])
            
    return np.array(X_lstm_list), np.array(X_cid_list), np.array(Y_list)

X_lstm_all, X_cid_all, Y_all = build_real_sequences(df, seq_len=SEQ_LEN)

if X_lstm_all.ndim < 3:
    raise ValueError(f"[ERROR] Tensor dimension mismatch. Shape: {X_lstm_all.shape}")

# Flatten the latest timestep for XGBoost 2D input
X_xgb_latest = X_lstm_all[:, -1, :]
X_xgb_all = np.hstack((X_xgb_latest, X_cid_all.reshape(-1, 1)))

print(f"       - LSTM Input Shape: {X_lstm_all.shape}")
print(f"       - XGBoost Input Shape: {X_xgb_all.shape}")


# 3. MODEL ARCHITECTURE DEFINITION
class GlobalConflictLSTM(nn.Module):
    def __init__(self, num_countries, embed_dim=16, input_dim=11, hidden_dim=128):
        super(GlobalConflictLSTM, self).__init__()
        self.country_embedding = nn.Embedding(num_countries, embed_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim + embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)
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


# 4. TIME SERIES CROSS-VALIDATION & ENSEMBLE EVALUATION
print("[INFO] Starting time-series cross-validation...")

n_total_samples = len(Y_all)
fold_step = n_total_samples // 4

for fold in range(1, 4):
    train_bound = fold * fold_step
    val_bound = train_bound + fold_step
    
    if train_bound >= n_total_samples or train_bound == 0:
        continue
        
    print(f"\n--- FOLD {fold} PROGRESS ---")
    
    X_xgb_train, X_xgb_val = X_xgb_all[:train_bound], X_xgb_all[train_bound:val_bound]
    X_lstm_train, X_lstm_val = X_lstm_all[:train_bound], X_lstm_all[train_bound:val_bound]
    X_cid_train, X_cid_val = X_cid_all[:train_bound], X_cid_all[train_bound:val_bound]
    y_train, y_val = Y_all[:train_bound], Y_all[train_bound:val_bound]
    
    if len(y_val) == 0 or len(np.unique(y_train)) < 2:
        print(f"[WARN] Fold {fold} skipped due to class imbalance or insufficient validation samples.")
        continue
        
    # (1) Train XGBoost Classifier
    xgb_global = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        subsample=0.8, 
        eval_metric="logloss", 
        random_state=42
    )
    xgb_global.fit(X_xgb_train, y_train)
    preds_xgb = xgb_global.predict_proba(X_xgb_val)[:, 1]
    
    # (2) Train PyTorch LSTM Model
    train_ds = TensorDataset(
        torch.FloatTensor(X_lstm_train), 
        torch.LongTensor(X_cid_train), 
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    lstm_global = GlobalConflictLSTM(num_countries=TOTAL_COUNTRIES, input_dim=N_FEATURES)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lstm_global.parameters(), lr=0.002)
    
    lstm_global.train()
    for epoch in range(10): 
        for seq_b, cid_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = lstm_global(seq_b, cid_b).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            
    lstm_global.eval()
    with torch.no_grad():
        preds_lstm = lstm_global(torch.FloatTensor(X_lstm_val), torch.LongTensor(X_cid_val)).squeeze().numpy()
        if preds_lstm.ndim == 0:
            preds_lstm = np.array([preds_lstm.item()])
        
    # (3) Soft Voting Ensemble Blending
    preds_ensemble = ALPHA * preds_xgb + (1 - ALPHA) * preds_lstm
    
    # (4) Performance Metrics Evaluation
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, preds_ensemble)
        prec, rec, thres = precision_recall_curve(y_val, preds_ensemble)
        f1_arr = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_arr)
        best_f1 = f1_arr[best_idx]
        best_th = thres[best_idx] if best_idx < len(thres) else 0.5
    else:
        auc = 0.5
        best_f1 = 0.0
        best_th = 0.5
        
    brier = brier_score_loss(y_val, preds_ensemble)
    
    print(f"  Evaluation Metrics (Fold {fold}):")
    print(f"    - AUC-ROC             : {auc:.4f}")
    print(f"    - Global Best F1 Score: {best_f1:.4f} (Threshold: {best_th:.3f})")
    print(f"    - Calibration (Brier) : {brier:.4f}")

print("\n[INFO] Global conflict ensemble pipeline execution completed successfully.")