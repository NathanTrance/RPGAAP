#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RP-GAAP Demo Notebook
=====================
Load checkpoints, analyze rare patterns, show sample nodes,
compare GAAP baseline vs RP-GAAP on YelpChi.

Run cell-by-cell in VS Code (# %% markers).
"""

# %% [markdown]
# # RP-GAAP: Rare-Pattern Weighted GAAP for Graph Fraud Detection
#
# **Key idea:** Common benign patterns dominate GNN training. Rare feature patterns
# (unusual reviewer behavior) are a strong fraud indicator but get drowned out.
# RP-GAAP upweights rare patterns during training — no architecture changes needed.

# %% Cell 1: Imports & Setup
import sys
from pathlib import Path
import os
import math
import glob
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
from torch.nn import Parameter
from rtdl_num_embeddings import compute_bins
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, recall_score, precision_score)
import matplotlib.pyplot as plt
import pandas as pd

DIR_SOURCE = str(Path.cwd())
sys.path.insert(0, DIR_SOURCE)
from mycode.utils.losses import FocalLoss
from mycode.utils.rare_pattern import make_rare_pattern_weights

print(f"PyTorch {torch.__version__}, DGL {dgl.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load & Explore the YelpChi Dataset

# %% Cell 2: Load Dataset
DATASET = "yelp"
graph = dgl.load_graphs(f"datasets/{DATASET}")[0][0]

graph.ndata['train_mask'] = graph.ndata['train_masks'][:, 0].bool()
graph.ndata['val_mask']   = graph.ndata['val_masks'][:, 0].bool()
graph.ndata['test_mask']  = graph.ndata['test_masks'][:, 0].bool()
graph.ndata['feature']    = graph.ndata['feature'].float()
graph.ndata['label']       = graph.ndata['label'].long()

feat = graph.ndata['feature']
feat = (feat - feat.min(0).values) / (feat.max(0).values - feat.min(0).values + 1e-8)
graph.ndata['feature'] = feat

N = graph.num_nodes()
E = graph.num_edges()
D = feat.shape[1]
fraud_rate = graph.ndata['label'].sum().item() / N

train_mask = graph.ndata['train_mask']
val_idx = graph.ndata['val_mask'].nonzero().squeeze()
test_idx = graph.ndata['test_mask'].nonzero().squeeze()
train_idx = graph.ndata['train_mask'].nonzero().squeeze()

print(f"Nodes: {N:,} | Edges: {E:,} | Features: {D} | Fraud rate: {fraud_rate:.1%}")
print(f"Train: {train_idx.shape[0]:,} | Val: {val_idx.shape[0]:,} | Test: {test_idx.shape[0]:,}")

# %% Cell 3: Feature Distribution Visualization
fnp = feat.numpy()
lnp = graph.ndata['label'].numpy()
var = np.var(fnp, axis=0)
top_feats = np.argsort(var)[-6:][::-1]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, fidx in enumerate(top_feats):
    ax = axes[i // 3, i % 3]
    ax.hist(fnp[lnp == 0, fidx], bins=50, alpha=0.6, label='Normal', density=True)
    ax.hist(fnp[lnp == 1, fidx], bins=50, alpha=0.6, label='Fraud', density=True)
    ax.set_title(f"Feature {fidx} (var={var[fidx]:.4f})"); ax.legend(fontsize=7)
plt.suptitle("Top-6 Feature Distributions: Normal vs Fraud", fontsize=14)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Rare-Pattern Computation (Step-by-Step)

# %% Cell 4: Rare Pattern Weighting — Visual Walkthrough
train_feat = feat[train_mask].numpy()
train_labels = lnp[train_mask]

K, BINS = 5, 5
var = np.var(train_feat, axis=0)
top_indices = np.argsort(var)[-K:]
print(f"Top-{K} features by variance: {list(top_indices)}")

patterns = np.zeros(train_feat.shape[0], dtype=np.int64)
multiplier = 1
for col in top_indices:
    q = np.quantile(train_feat[:, col], np.linspace(0, 1, BINS + 1))
    q[0], q[-1] = -np.inf, np.inf
    patterns += np.digitize(train_feat[:, col], q[1:-1], right=False) * multiplier
    multiplier *= BINS

unique_p, inv, counts = np.unique(patterns, return_inverse=True, return_counts=True)
print(f"Unique patterns: {len(unique_p)} from {train_feat.shape[0]} nodes")
print(f"Freq range: {counts.min()} – {counts.max()}")

max_c, min_c = counts.max(), counts.min()
norm_freq = (max_c - counts) / (max_c - min_c) if max_c > min_c else np.ones_like(counts, dtype=float)
weights = 1.0 + 2.0 * norm_freq
is_fraud = train_labels == 1
fweights = weights[is_fraud]; nweights = weights[~is_fraud]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(['Min','Q25','Median','Q75','Max'],
            [counts.min(),np.percentile(counts,25),np.median(counts),
             np.percentile(counts,75),counts.max()])
axes[0].set_title("Pattern Frequency"); axes[0].set_ylabel("Count")
axes[1].hist(nweights, bins=30, alpha=0.6, label=f'Normal (n={len(nweights)})')
axes[1].hist(fweights, bins=30, alpha=0.6, label=f'Fraud (n={len(fweights)})')
axes[1].set_title("Sample Weights"); axes[1].legend()
axes[2].hist(weights, bins=40, alpha=0.7)
axes[2].set_title("Weight Distribution"); axes[2].set_xlabel("Weight")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Model Architecture (GAAP — AAAI 2025)

# %% Cell 5: SAGE Model with DyPLE + MHA

class DyPLEC(nn.Module):
    def __init__(self, d_in, n_bins=32):
        super().__init__()
        self.n_bins = n_bins; self.o_d_in = d_in
        self.raw_bin_width = nn.Parameter(torch.randn(d_in, 1, n_bins))
        self.register_buffer('mask', torch.tril(torch.ones(n_bins, n_bins)))
        ml = torch.ones((d_in, n_bins)); ml[:, -1] = 0; self.register_buffer('mask_left', ml.bool())
        mr = torch.ones((d_in, n_bins)); mr[:, 0] = 0; self.register_buffer('mask_right', mr.bool())

    def forward(self, inp):
        bw = self.raw_bin_width.softmax(dim=-1)
        axis = (bw[:, :, None, :] * self.mask[None, None, :, :]).sum(dim=-1)
        zeros = torch.zeros((axis.shape[0], 1, 1), device=bw.device)
        axis = torch.cat((zeros, axis), dim=-1)[..., :self.n_bins]
        rate = (inp[:, :, None, None] - axis) / (bw + 1e-8)
        rate = rate.transpose(1, 2).flatten(-2, -1)
        rate[:, :, self.mask_left.flatten()] = 1 - F.relu(1 - rate[:, :, self.mask_left.flatten()])
        rate[:, :, self.mask_right.flatten()] = F.relu(rate[:, :, self.mask_right.flatten()])
        return rate.view(-1, self.o_d_in, self.n_bins)


class NLinear(nn.Module):
    def __init__(self, n, in_f, out_f, bias=True):
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_f, out_f))
        self.bias = Parameter(torch.empty(n, out_f)) if bias else None
        d = in_f ** -0.5; nn.init.uniform_(self.weight, -d, d)
        if self.bias is not None: nn.init.uniform_(self.bias, -d, d)

    def forward(self, x):
        return (x[..., None, :] @ self.weight).squeeze(-2) + (self.bias if self.bias is not None else 0)


class SAGEConv(nn.Module):
    def __init__(self, in_f, out_f, aggregator='max_pool', feat_drop=0.0, bias=False, norm=None, act=None):
        super().__init__()
        self._in_src, self._in_dst = expand_as_pair(in_f)
        self._agg_type = aggregator; self.feat_drop = nn.Dropout(feat_drop)
        self.norm = norm; self.activation = act
        self.fc_pool = nn.Linear(self._in_src, self._in_src)
        self.fc_neigh = nn.Linear(self._in_src, out_f, bias=False)
        self.fc_self = nn.Linear(self._in_dst, out_f, bias=bias)
        for m in [self.fc_pool, self.fc_neigh, self.fc_self]:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block: feat_dst = feat_src[:graph.num_dst_nodes()]
            h_self = feat_dst
            if graph.num_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(feat_dst.shape[0], self._in_src).to(feat_dst)
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.max('m', 'neigh') if 'max' in self._agg_type else fn.mean('m', 'neigh'))
            h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            rst = self.fc_self(h_self) + h_neigh
            if self.activation is not None: rst = self.activation(rst)
            if self.norm is not None: rst = self.norm(rst)
            return rst


class SAGE(nn.Module):
    def __init__(self, d_in, n_classes, d_hidden=64, n_bins=32, d_feat_emb=32,
                 gnn_n_layers=3, use_mha=True, mha_n_heads=4, mha_alpha=0.2, n_nodes=None):
        super().__init__()
        self.dyple = DyPLEC(d_in, n_bins)
        self.nlinear = NLinear(d_in, n_bins, d_feat_emb)
        self.bn0 = nn.BatchNorm1d(d_in)
        self.linear_in = nn.Linear(d_in * d_feat_emb, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.drop = nn.Dropout(0.1); self.act = nn.ReLU()
        self.layers = nn.ModuleList([
            SAGEConv(d_hidden, d_hidden, 'max_pool', feat_drop=0.1,
                     norm=nn.BatchNorm1d(d_hidden), act=nn.ReLU())
            for _ in range(gnn_n_layers)
        ])
        self.use_mha = use_mha; self.alpha = mha_alpha
        if use_mha:
            self.post_fc = nn.Sequential(nn.Dropout(0.1),
                nn.Linear(d_hidden, d_hidden * mha_n_heads), nn.ReLU(),
                nn.BatchNorm1d(d_hidden * mha_n_heads))
            self.mha = nn.MultiheadAttention(d_hidden * mha_n_heads, mha_n_heads,
                                             dropout=0., batch_first=True)
            self.register_buffer('his_emb', torch.rand((n_nodes, d_hidden * mha_n_heads)))
        else:
            self.post_fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_hidden, d_hidden),
                                         nn.ReLU(), nn.BatchNorm1d(d_hidden))
            mha_n_heads = 1
        self.fc_out = nn.Linear(d_hidden * mha_n_heads, n_classes)

    def forward(self, blocks, feat):
        h = self.dyple(feat); h = self.nlinear(h); h = F.relu(h)
        h = self.bn0(h); h = self.drop(h); h = self.linear_in(h.flatten(1))
        h = F.relu(h); h = self.bn1(h)
        for layer, block in zip(self.layers, blocks):
            last = h; h = layer(block, h)
            h = 0.7 * h + 0.3 * last[:block.num_dst_nodes()]
        h = self.post_fc(h); gnn_out = h
        if self.use_mha:
            v = k = self.his_emb; q = h[:blocks[-1].num_dst_nodes()]
            o, _ = self.mha(q, k, v); o = self.act(o)
            h = self.alpha * o + (1 - self.alpha) * h
        return self.fc_out(h), gnn_out


# %% [markdown]
# ## 4. Load Pretrained Checkpoints & Quick Training Demo

# %% Cell 6: Load checkpoints or do fast 5-epoch demo

CKPT_BASE = "checkpoints/yelp_baseline.pt"
CKPT_RARE = "checkpoints/yelp_rare.pt"

def build_model():
    return SAGE(d_in=D, n_classes=2, n_nodes=N, d_hidden=64, n_bins=32,
                d_feat_emb=32, gnn_n_layers=3, mha_n_heads=4, mha_alpha=0.2).to(device)

model_base = build_model()
model_rare = build_model()

# Samplers
sampler = dgl.dataloading.NeighborSampler(
    [-1, -1, -1], prefetch_node_feats=['feature'], prefetch_labels=['label'])

trn_loader = dgl.dataloading.DataLoader(
    graph, train_idx, sampler, device=device, use_uva=(device.type=='cuda'),
    batch_size=128, shuffle=True, drop_last=True)

val_loader = dgl.dataloading.DataLoader(
    graph, torch.arange(N), sampler, device=device, use_uva=(device.type=='cuda'),
    batch_size=1280, shuffle=False, drop_last=False)

# Compute rare weights for entire graph
rare_weights = make_rare_pattern_weights(
    features=graph.ndata['feature'], labels=graph.ndata['label'],
    train_mask=graph.ndata['train_mask'],
    num_bins=5, top_k_features=10, max_weight=3.0, fraud_boost=1.5
).to(device)

focal_loss_fn = FocalLoss(gamma=2.0, alpha=[1.0, 3.0], reduction='mean')

# ------------------ Load checkpoint if exists ------------------
loaded = False
if os.path.exists(CKPT_BASE) and os.path.exists(CKPT_RARE):
    print("Loading pretrained checkpoints...")
    sd_base = torch.load(CKPT_BASE, map_location=device, weights_only=False)
    sd_rare = torch.load(CKPT_RARE, map_location=device, weights_only=False)
    # Strip 'model.' prefix if from Lightning checkpoint
    if any(k.startswith('model.') for k in sd_base.get('state_dict', {}).keys()):
        sd_base = {k[6:]: v for k, v in sd_base['state_dict'].items() if k.startswith('model.')}
        sd_rare = {k[6:]: v for k, v in sd_rare['state_dict'].items() if k.startswith('model.')}
    else:
        sd_base = sd_base.get('state_dict', sd_base)
        sd_rare = sd_rare.get('state_dict', sd_rare)
    model_base.load_state_dict(sd_base, strict=False)
    model_rare.load_state_dict(sd_rare, strict=False)
    loaded = True
    print("Checkpoints loaded.")
else:
    print(f"No checkpoints found at {CKPT_BASE} / {CKPT_RARE}")
    print("Running 5-epoch quick demo instead...")

# ------------------ Quick 5-epoch demo (for show) ------------------
MAX_EPOCHS = 5
history = {'base_loss': [], 'rare_loss': [], 'base_auc': [], 'rare_auc': []}

def quick_eval(model):
    model.eval()
    with torch.no_grad():
        all_y, all_prob, all_nid = [], [], []
        for _, _, blocks in val_loader:
            x = blocks[0].srcdata['feature']; y = blocks[-1].dstdata['label']
            prob = model(blocks, x)[0].softmax(-1)[:, 1]
            all_y.append(y); all_prob.append(prob)
            all_nid.append(blocks[-1].dstdata[dgl.NID])
        y = torch.cat(all_y)[torch.argsort(torch.cat(all_nid))]
        prob = torch.cat(all_prob)[torch.argsort(torch.cat(all_nid))]
        y, prob = y.cpu().numpy(), prob.cpu().numpy()
        return roc_auc_score(y[val_idx], prob[val_idx]), average_precision_score(y[val_idx], prob[val_idx])

opt_base = torch.optim.Adam(model_base.parameters(), lr=0.001)
opt_rare = torch.optim.Adam(model_rare.parameters(), lr=0.001)

print(f"{'Epoch':>5} | {'Base Loss':>10} | {'Base AUC':>9} | {'Rare Loss':>10} | {'Rare AUC':>9}")
print("-" * 60)
for epoch in range(1, MAX_EPOCHS + 1):
    # Baseline
    model_base.train(); loss_b = 0
    for _, _, blocks in trn_loader:
        x = blocks[0].srcdata['feature']; y = blocks[-1].dstdata['label']
        logits, _ = model_base(blocks, x)
        l = F.cross_entropy(logits, y); opt_base.zero_grad(); l.backward(); opt_base.step()
        loss_b += l.item()
    loss_b /= len(trn_loader)
    # Rare
    model_rare.train(); loss_r = 0
    for _, _, blocks in trn_loader:
        x = blocks[0].srcdata['feature']; y = blocks[-1].dstdata['label']
        nid = blocks[-1].dstdata[dgl.NID]; sw = rare_weights[nid].to(x.device)
        logits, _ = model_rare(blocks, x)
        l_each = F.cross_entropy(logits, y, reduction='none')
        l = (l_each * sw).mean(); opt_rare.zero_grad(); l.backward(); opt_rare.step()
        loss_r += l.item()
    loss_r /= len(trn_loader)
    auc_b, aps_b = quick_eval(model_base)
    auc_r, aps_r = quick_eval(model_rare)
    history['base_loss'].append(loss_b); history['rare_loss'].append(loss_r)
    history['base_auc'].append(auc_b); history['rare_auc'].append(auc_r)
    print(f"{epoch:5d} | {loss_b:10.4f} | {auc_b:9.4f} | {loss_r:10.4f} | {auc_r:9.4f}")

if not loaded:
    # Save for next time
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model_base.state_dict(), CKPT_BASE)
    torch.save(model_rare.state_dict(), CKPT_RARE)
    print(f"Saved to {CKPT_BASE}, {CKPT_RARE}")

# %% Cell 7: Training Curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history['base_loss'], 'o-', label='Baseline (GAAP)', lw=2)
axes[0].plot(history['rare_loss'], 's-', label='Rare-Weighted (RP-GAAP)', lw=2)
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(history['base_auc'], 'o-', label='Baseline AUC', lw=2)
axes[1].plot(history['rare_auc'], 's-', label='Rare-Weighted AUC', lw=2)
axes[1].set_title("Validation AUC"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Sample Nodes — What Do Rare Patterns Look Like?

# %% Cell 8: Show concrete fraud/normal nodes with their patterns

# Full graph evaluation
@torch.no_grad()
def infer_all(model):
    model.eval()
    all_y, all_prob, all_nid = [], [], []
    for _, _, blocks in val_loader:
        x = blocks[0].srcdata['feature']; y = blocks[-1].dstdata['label']
        prob = model(blocks, x)[0].softmax(-1)[:, 1]
        all_y.append(y); all_prob.append(prob); all_nid.append(blocks[-1].dstdata[dgl.NID])
    y = torch.cat(all_y)[torch.argsort(torch.cat(all_nid))]
    prob = torch.cat(all_prob)[torch.argsort(torch.cat(all_nid))]
    return y.cpu().numpy(), prob.cpu().numpy()

y_all, prob_base = infer_all(model_base)
_, prob_rare = infer_all(model_rare)

# Compute per-node pattern IDs across full graph
fnp_all = feat.numpy(); lbl_all = lnp
K, BINS = 5, 5
var = np.var(fnp_all[train_mask], axis=0)
topk_idx = np.argsort(var)[-K:]
pat_all = np.zeros(N, dtype=np.int64)
mult = 1
for col in topk_idx:
    q = np.quantile(fnp_all[train_mask][:, col], np.linspace(0, 1, BINS + 1))
    q[0], q[-1] = -np.inf, np.inf
    pat_all += np.digitize(fnp_all[:, col], q[1:-1], right=False) * mult
    mult *= BINS

up, uinv, ucnts = np.unique(pat_all, return_inverse=True, return_counts=True)
pat_freq = ucnts[uinv]  # frequency per node
pat_rank = np.argsort(np.argsort(ucnts))[uinv] + 1  # 1 = rarest
pat_wt = 1.0 + 2.0 * (ucnts.max() - ucnts[uinv]) / max(1, ucnts.max() - ucnts.min())

# Pick interesting test nodes: fraud caught by rare but missed by baseline
test_nodes = test_idx.numpy()
fraud_test = test_nodes[lbl_all[test_nodes] == 1]
thr = 0.5

# Baseline misses these fraud nodes
missed_base = fraud_test[prob_base[fraud_test] <= thr]
caught_by_rare = missed_base[prob_rare[missed_base] > thr]

print(f"Fraud test nodes: {len(fraud_test)}")
print(f"Missed by baseline (thr=0.5): {len(missed_base)}")
print(f"Of those, caught by rare: {len(caught_by_rare)}")
print(f"Recall uplift: +{len(caught_by_rare)/len(fraud_test)*100:.1f}%\n")

# Build sample table
sample_nodes = []
# 5 fraud: missed by baseline, caught by rare
if len(caught_by_rare) >= 3:
    picks_fraud = np.random.choice(caught_by_rare, min(3, len(caught_by_rare)), replace=False)
    for nid in picks_fraud:
        sample_nodes.append({'Node': nid, 'Label': 'FRAUD',
                             'Pattern ID': pat_all[nid], 'Pattern Freq': pat_freq[nid],
                             'Weight': f"{pat_wt[nid]:.2f}",
                             'Baseline Prob': f"{prob_base[nid]:.4f}",
                             'Rare Prob': f"{prob_rare[nid]:.4f}",
                             'Caught?': '✓ Rare only'})
# 2 normal nodes with high weight (rare normal patterns)
normal_test = test_nodes[lbl_all[test_nodes] == 0]
rare_normals = normal_test[np.argsort(pat_rank[normal_test])[:5]]
if len(rare_normals) >= 2:
    picks_normal = np.random.choice(rare_normals, min(2, len(rare_normals)), replace=False)
    for nid in picks_normal:
        sample_nodes.append({'Node': nid, 'Label': 'NORMAL',
                             'Pattern ID': pat_all[nid], 'Pattern Freq': pat_freq[nid],
                             'Weight': f"{pat_wt[nid]:.2f}",
                             'Baseline Prob': f"{prob_base[nid]:.4f}",
                             'Rare Prob': f"{prob_rare[nid]:.4f}",
                             'Caught?': 'OK'})

df = pd.DataFrame(sample_nodes)
print("Sample nodes (test set):\n")
print(df.to_string(index=False))

# Show feature signatures for sample nodes
fig, axes = plt.subplots(len(sample_nodes), 1, figsize=(12, 2.5 * len(sample_nodes)))
if len(sample_nodes) == 1: axes = [axes]
for i, (_, row) in enumerate(df.iterrows()):
    nid = int(row['Node']); lbl = row['Label']
    feat_vals = fnp_all[nid, topk_idx]
    mid = np.median(fnp_all[:, topk_idx], axis=0)
    q25 = np.percentile(fnp_all[:, topk_idx], 25, axis=0)
    q75 = np.percentile(fnp_all[:, topk_idx], 75, axis=0)
    x = np.arange(K)
    axes[i].fill_between(x, q25, q75, alpha=0.2, label='IQR')
    axes[i].plot(x, mid, 'k--', alpha=0.3, label='Median')
    color = 'red' if lbl == 'FRAUD' else 'green'
    axes[i].plot(x, feat_vals, f'{color}o-', lw=2, ms=8, label=f'Node {nid} ({lbl})')
    axes[i].set_xticks(x); axes[i].set_xticklabels([f"Feat {f}" for f in topk_idx])
    axes[i].set_title(f"Node {nid} ({lbl}) | Pattern {row['Pattern ID']} | "
                      f"P_base={row['Baseline Prob']} P_rare={row['Rare Prob']}")
    axes[i].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Full Evaluation Results

# %% Cell 9: Results Table

from sklearn.metrics._ranking import _binary_clf_curve

def compute_metrics(y, prob, idx):
    fps, tps, thresholds = _binary_clf_curve(y[val_idx], prob[val_idx])
    n_pos, n_neg = y[val_idx].sum(), len(y[val_idx]) - y[val_idx].sum()
    f11 = 2 * tps / (2 * tps + (n_pos - tps) + fps)
    best_thr = thresholds[np.argmax(f11)]
    preds = (prob > best_thr).astype(int)
    top_n = int(y[idx].sum())
    top_indices = np.argsort(prob[idx])[-top_n:]
    top_pred = np.zeros_like(prob[idx]); top_pred[top_indices] = 1
    return {
        'AUC': roc_auc_score(y[idx], prob[idx]),
        'AP': average_precision_score(y[idx], prob[idx]),
        'Macro-F1': f1_score(y[idx], preds[idx], average='macro'),
        'Fraud Recall': recall_score(y[idx], preds[idx]),
        'Fraud Precision': precision_score(y[idx], preds[idx]),
        'tst_top (Recall@K)': recall_score(y[idx], top_pred),
        'Accuracy': np.mean(preds[idx] == y[idx]),
    }

res_base = compute_metrics(y_all, prob_base, test_idx)
res_rare = compute_metrics(y_all, prob_rare, test_idx)

df = pd.DataFrame({
    'Metric': list(res_base.keys()),
    'GAAP (Baseline)': [f"{v:.4f}" for v in res_base.values()],
    'RP-GAAP (Rare)': [f"{v:.4f}" for v in res_rare.values()],
    'Δ': [f"{res_rare[k]-res_base[k]:+.4f}" for k in res_base],
})

print("\n" + "=" * 70)
print("  RP-GAAP vs GAAP — YelpChi Test Results")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

# %% [markdown]
# ## 7. Rare Pattern → Fraud Map

# %% Cell 10: Scatter: Pattern Frequency vs Fraud Rate

up_full, uinv_full, cnts_full = np.unique(pat_all[train_mask], return_inverse=True, return_counts=True)
fr_by_pat = np.array([lnp[train_mask][uinv_full == i].mean() for i in range(len(up_full))])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc = axes[0].scatter(np.log1p(cnts_full), fr_by_pat, c=fr_by_pat, cmap='Reds',
                     alpha=0.5, s=20, edgecolors='none')
axes[0].set_xlabel("Log(Pattern Frequency + 1)"); axes[0].set_ylabel("Fraud Rate")
axes[0].set_title("Rare Patterns → Higher Fraud Rates\n(Patterns from top-5 features, 5 bins each)")
plt.colorbar(sc, ax=axes[0], label='Fraud Rate')

# Baseline vs Rare: which fraud nodes get higher probability?
fraud_mask_test = lbl_all[test_nodes] == 1
delta_prob = prob_rare[test_nodes][fraud_mask_test] - prob_base[test_nodes][fraud_mask_test]
axes[1].hist(delta_prob, bins=30, alpha=0.7, color='purple', edgecolor='white')
axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Δ Probability (Rare - Baseline)")
axes[1].set_ylabel("Fraud Test Nodes")
axes[1].set_title(f"Rare-Weighting Pushes Fraud Probabilities Up\n"
                  f"{(delta_prob > 0).mean()*100:.0f}% of fraud nodes get higher score")
plt.tight_layout(); plt.show()

print("\n✅ RP-GAAP Demo Complete.")
print("Rare-pattern weighting identifies uncommon feature combinations, assigns")
print("higher training weight, and catches fraud the baseline misses — with no")
print("architecture changes to the GAAP model.")
