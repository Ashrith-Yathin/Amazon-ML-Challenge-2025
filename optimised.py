# OPTIMIZED MLP FUSION FOR TEXT + IMAGE EMBEDDINGS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import gc
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MLP FUSION: Text + Image Embeddings")
print("="*70)

# ADVANCED MLP ARCHITECTURE

class MultimodalFusionMLP(nn.Module):
    """
    Advanced fusion with separate encoders and attention.
    """
    def __init__(self, text_dim, image_dim, other_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.other_encoder = nn.Sequential( # Encoder for quantity, brand, etc.
            nn.Linear(other_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout * 0.5)
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        # Fusion layers to combine all encoded parts
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def forward(self, text_emb, image_emb, other_emb):
        text_enc = self.text_encoder(text_emb)
        image_enc = self.image_encoder(image_emb)
        other_enc = self.other_encoder(other_emb)
        
        # Cross-attention: text attends to image
        attended, _ = self.attention(text_enc.unsqueeze(1), image_enc.unsqueeze(1), image_enc.unsqueeze(1))
        attended = attended.squeeze(1)
        
        # Concatenate attended text, original image, and other features
        fused = torch.cat([attended, image_enc, other_enc], dim=1)
        output = self.fusion(fused)
        return output

# TRAINING FUNCTION (UPDATED FOR 3 INPUTS)

def train_mlp_fusion(X_text_tr, X_image_tr, X_other_tr, y_tr, 
                     X_text_val, X_image_val, X_other_val, y_val,
                     epochs=100, batch_size=256, lr=5e-4):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    text_dim, image_dim, other_dim = X_text_tr.shape[1], X_image_tr.shape[1], X_other_tr.shape[1]
    
    model = MultimodalFusionMLP(text_dim, image_dim, other_dim).to(device)
    
    def pseudo_huber_loss(pred, target, delta=1.0):
        residual = pred - target
        return torch.mean(delta**2 * (torch.sqrt(1 + (residual/delta)**2) - 1))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_text_tr), torch.FloatTensor(X_image_tr), torch.FloatTensor(X_other_tr), torch.FloatTensor(y_tr).unsqueeze(1))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_text_val), torch.FloatTensor(X_image_val), torch.FloatTensor(X_other_val), torch.FloatTensor(y_val).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf'); patience, patience_counter = 15, 0
    
    for epoch in range(epochs):
        model.train(); train_loss = 0
        for text_b, image_b, other_b, y_b in train_loader:
            text_b, image_b, other_b, y_b = text_b.to(device), image_b.to(device), other_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            output = model(text_b, image_b, other_b)
            loss = pseudo_huber_loss(output, y_b)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()
        
        model.eval(); val_loss = 0
        with torch.no_grad():
            for text_b, image_b, other_b, y_b in val_loader:
                text_b, image_b, other_b, y_b = text_b.to(device), image_b.to(device), other_b.to(device), y_b.to(device)
                output = model(text_b, image_b, other_b)
                val_loss += pseudo_huber_loss(output, y_b).item()
        
        train_loss /= len(train_loader); val_loss /= len(val_loader); scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss, best_model_state, patience_counter = val_loss, model.state_dict(), 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience: print(f"    Early stopping at epoch {epoch+1}"); break
        if (epoch + 1) % 10 == 0: print(f"    Epoch {epoch+1}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
            
    model.load_state_dict(best_model_state)
    return model

# PREDICTION FUNCTION (UPDATED FOR 3 INPUTS)

def predict_mlp(model, X_text, X_image, X_other, batch_size=256):
    device = next(model.parameters()).device
    model.eval(); predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_text), batch_size), desc="Predicting", leave=False):
            end_idx = min(i + batch_size, len(X_text))
            text_b = torch.FloatTensor(X_text[i:end_idx]).to(device)
            image_b = torch.FloatTensor(X_image[i:end_idx]).to(device)
            other_b = torch.FloatTensor(X_other[i:end_idx]).to(device)
            output = model(text_b, image_b, other_b)
            predictions.append(output.cpu().numpy())
    return np.vstack(predictions).flatten()

# CORRECTED DATA LOADING AND SLICING 
print("\n[1/4] Loading and slicing combined embeddings...")
df_train = pd.read_csv('train.csv')
y_train_log = np.log1p(df_train['price'].values)

# Load the SINGLE, COMBINED feature files , use correct paths
X_train_full = np.load("final_X_train_medium_with_brand.npy", allow_pickle=False)
X_test_full = np.load("final_X_test_medium_with_brand.npy", allow_pickle=False)

# Define the dimensions of your features
text_dim = 384 # From SentenceTransformer
image_dim = 512 # From ViT-B/16

# Slice the combined arrays into their constituent parts
train_text = X_train_full[:, :text_dim]
train_image = X_train_full[:, text_dim:text_dim+image_dim]
train_other = X_train_full[:, text_dim+image_dim:]

test_text = X_test_full[:, :text_dim]
test_image = X_test_full[:, text_dim:text_dim+image_dim]
test_other = X_test_full[:, text_dim+image_dim:]

print(f"✓ Text: train{train_text.shape}, test{test_text.shape}")
print(f"✓ Image: train{train_image.shape}, test{test_image.shape}")
print(f"✓ Other: train{train_other.shape}, test{test_other.shape}")
del X_train_full, X_test_full; gc.collect()

# SCALE FEATURES

print("\n[2/4] Scaling features...")
text_scaler, image_scaler, other_scaler = RobustScaler(), RobustScaler(), RobustScaler()

train_text_scaled = text_scaler.fit_transform(train_text); test_text_scaled = text_scaler.transform(test_text)
train_image_scaled = image_scaler.fit_transform(train_image); test_image_scaled = image_scaler.transform(test_image)
train_other_scaled = other_scaler.fit_transform(train_other); test_other_scaled = other_scaler.transform(test_other)

print("✓ Features scaled")
del train_text, train_image, train_other, test_text, test_image, test_other; gc.collect()

# K-FOLD TRAINING

print("\n[3/4] Training MLP with K-Fold...")
N_FOLDS = 5; kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_text_scaled)); test_preds = np.zeros(len(test_text_scaled))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_text_scaled), 1):
    print(f"\n{'─'*70}\nFOLD {fold}/{N_FOLDS}")
    
    model = train_mlp_fusion(
        train_text_scaled[train_idx], train_image_scaled[train_idx], train_other_scaled[train_idx], y_train_log[train_idx],
        train_text_scaled[val_idx], train_image_scaled[val_idx], train_other_scaled[val_idx], y_train_log[val_idx]
    )
    
    oof_preds[val_idx] = predict_mlp(model, train_text_scaled[val_idx], train_image_scaled[val_idx], train_other_scaled[val_idx])
    test_preds += predict_mlp(model, test_text_scaled, test_image_scaled, test_other_scaled) / N_FOLDS
    
    val_pred_price = np.expm1(oof_preds[val_idx]); val_actual_price = np.expm1(y_train_log[val_idx])
    fold_smape = np.mean(2 * np.abs(val_pred_price - val_actual_price) / (np.abs(val_actual_price) + np.abs(val_pred_price) + 1e-8)) * 100
    print(f"Fold {fold} SMAPE: {fold_smape:.4f}%")
    
    del model; gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None


# FINAL EVALUATION

print("\n[4/4] Final evaluation and submission...")
oof_prices = np.expm1(oof_preds); actual_prices = df_train['price'].values
overall_smape = np.mean(2 * np.abs(oof_prices - actual_prices) / (np.abs(actual_prices) + np.abs(oof_prices) + 1e-8)) * 100
print("\n" + "="*70 + f"\nFINAL OOF SMAPE: {overall_smape:.4f}%\n" + "="*70)

final_predictions = np.expm1(test_preds); final_predictions = np.clip(final_predictions, 0.01, None)
df_test = pd.read_csv('test.csv')
submission = pd.DataFrame({'sample_id': df_test['sample_id'],'price': final_predictions})
submission.to_csv('test_out.csv', index=False)

print("\nSubmission created: test_out.csv")
print("\nFirst 10 predictions:"); print(submission.head(10))
