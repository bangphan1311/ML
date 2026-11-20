
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

LOG_COLS = ["amount_usd", "account_balance_usd",
            "hours_since_last_txn", "distance_from_home_km"]
NUM_COLS = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes",
            "card_age_months", "customer_age", "velocity_score",
            "merchant_risk_score"]
CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel",
            "device_type", "time_of_day_hour", "day_of_week"]
BOOL_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch",
             "is_ai_generated_scam_attempt"]


class NNPreprocessor:
    def __init__(self, use_log=False):
        self.use_log = use_log

    def _numeric(self, df):
        x = df[LOG_COLS + NUM_COLS].astype("float64").copy()
        if self.use_log:
            x[LOG_COLS] = np.log1p(x[LOG_COLS])
        return x

    def fit(self, train_df):
        # Mọi thống kê chỉ học từ train của fold
        self.scaler = StandardScaler().fit(self._numeric(train_df))
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit(train_df[CAT_COLS])
        return self

    def transform(self, df):
        x_num = self.scaler.transform(self._numeric(df)).astype("float32")
        x_bool = df[BOOL_COLS].astype("float32").to_numpy()
        x_cat = (self.encoder.transform(df[CAT_COLS]) + 1).astype("int64")
        return np.hstack([x_num, x_bool]), x_cat


class TabMLP(nn.Module):
    def __init__(self, n_cont, cat_sizes, hidden=(256, 128), drop=0.2):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(n, min(8, (n + 1) // 2)) for n in cat_sizes])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        layers, d = [], n_cont + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat([x_cont] + e, dim=1)).squeeze(1)


def predict(model, Xn, Xc, device, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xb = torch.from_numpy(Xn[i:i + bs]).to(device)
            cb = torch.from_numpy(Xc[i:i + bs]).to(device)
            out.append(torch.sigmoid(model(xb, cb)).cpu().numpy())
    return np.concatenate(out)


def run_fold(train, k, device, epochs=8, lr=1e-3, bs=2048, seed=0, use_log=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = train[train.fold != k], train[train.fold == k]

    pre = NNPreprocessor(use_log=use_log).fit(tr)  # chỉ fit trên train của fold
    Xn_tr, Xc_tr = pre.transform(tr)
    Xn_va, Xc_va = pre.transform(va)
    y_tr = tr["is_fraud"].to_numpy("float32")
    y_va = va["is_fraud"].to_numpy()

    cat_sizes = [len(c) + 1 for c in pre.encoder.categories_]
    model = TabMLP(Xn_tr.shape[1], cat_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xn_tr), torch.from_numpy(Xc_tr),
                      torch.from_numpy(y_tr)),
        batch_size=bs, shuffle=True)

    t0, history = time.time(), []
    for ep in range(epochs):
        model.train()
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb, cb), yb).backward()
            opt.step()
        p = predict(model, Xn_va, Xc_va, device)
        auc = roc_auc_score(y_va, p)
        history.append(auc)
        print(f"fold {k} | epoch {ep + 1}/{epochs} | val ROC-AUC = {auc:.5f}", flush=True)
    return p, va.index, history, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--use_log", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="chạy thử nhanh")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    train = pd.read_csv(a.train)

    if a.smoke:
        train = train.sample(50_000, random_state=0)
        folds, epochs = [0], 1
    else:
        folds, epochs = range(5), a.epochs

    oof = pd.Series(np.nan, index=train.index)
    rows = []
    for k in folds:
        p, idx, hist, secs = run_fold(train, k, device, epochs=epochs, use_log=a.use_log)
        oof.loc[idx] = p
        rows.append({"fold": k, "roc_auc": hist[-1], "seconds": round(secs, 1)})
    print(pd.DataFrame(rows))

    if not a.smoke:
        print("OOF ROC-AUC:", roc_auc_score(train["is_fraud"], oof))
        pd.DataFrame({"id": train["id"], "oof_nn": oof}).to_csv("oof_nn.csv", index=False)
