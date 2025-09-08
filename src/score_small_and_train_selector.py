#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small sample scoring (v3.0 - ΔNLL replaces KL + soft gating + dynamic thresholds)
"""
import os
import json
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import random

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def _load_model(name, q4=False):
    """Load single model"""
    tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    kw = dict(device_map="auto", trust_remote_code=True)
    if q4:
        try:
            import bitsandbytes as bnb
            kw.update(dict(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ))
        except ImportError:
            print("Warning: bitsandbytes not installed, using bf16 instead of 4bit")
            kw.update(dict(torch_dtype=torch.bfloat16))
    else:
        kw.update(dict(torch_dtype=torch.bfloat16))
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, attn_implementation="flash_attention_2", **kw
        ).eval()
        print(f"  Loaded with Flash Attention 2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
        print(f"  Loaded with standard attention")
    
    return tok, model

def load_models(teacher_names, student_name, q4=False):
    """Load all models"""
    teachers = []
    for n in teacher_names:
        print(f"[load] teacher: {n}")
        T_tok, T = _load_model(n, q4=q4)
        teachers.append((T_tok, T))
    
    print(f"[load] student: {student_name}")
    S_tok, S = _load_model(student_name, q4=False)
    
    return teachers, (S_tok, S)

@torch.inference_mode()
def ppl_models(texts, tokenizers_models, ctx=1536, bs=16):
    """Per-sample PPL (more accurate)"""
    all_ppls = []
    
    for tok, model in tokenizers_models:
        ppls = []
        pbar = tqdm(range(0, len(texts), bs), desc=f"PPL@{ctx}", leave=False)
        
        for i in pbar:
            batch = texts[i:i+bs]
            enc = tok(batch, return_tensors="pt", truncation=True, 
                     max_length=ctx, padding=True)
            ids = enc.input_ids.to(model.device)
            attn = enc.attention_mask.to(model.device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=attn).logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            shift_mask = attn[:, 1:].contiguous().float()
            
            # Per-token NLL
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            nll_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(ids.size(0), -1)
            
            # Per-sample average NLL
            nll_sum = (nll_tok * shift_mask).sum(dim=1)
            tok_cnt = shift_mask.sum(dim=1).clamp_min(1.0)
            nll_mean = (nll_sum / tok_cnt).detach().float().cpu().numpy()
            
            # Convert to PPL with overflow protection
            sample_ppls = np.exp(np.minimum(nll_mean, 10.0))
            ppls.extend(sample_ppls)
            
            pbar.set_postfix({"avg_ppl": f"{np.mean(sample_ppls):.2f}"})
        
        all_ppls.append(np.array(ppls, dtype=np.float32))
    
    return np.mean(all_ppls, axis=0) if len(all_ppls) > 1 else all_ppls[0]

def format_bonus(r):
    """Calculate structured rewards (enhanced version)"""
    b = 0.0
    
    # Function calling reward (increased weight)
    if r.get("json_ok"): 
        b += 0.6
    if "function_calling" in r["tags"]: 
        b += 0.4
    if r.get("fc_sig"): 
        b += 0.3
    
    # Reasoning reward
    if "reasoning" in r["tags"]: 
        b += 0.3
    
    # Roleplay reward (increased weight)
    if "roleplay" in r["tags"]: 
        b += 0.25
    
    # ChatRAG reward (increased weight)
    if "chatrag" in r["tags"] or r.get("has_cite"): 
        b += 0.35
    
    # Fine-grained features
    b += 0.04 * min(6, r.get("json_depth", 0))
    b += 0.03 * min(12, r.get("json_keys", 0))
    b += 0.02 * min(10, r.get("fc_params", 0))
    b += 0.03 * min(12, r.get("steps", 0))
    b += 0.025 * min(12, r.get("turns", 0))
    
    return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fin", required=True)
    ap.add_argument("--out", dest="fout", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen2-7B")
    ap.add_argument("--student", type=str, default="data4elm/Llama-400M-12L")
    ap.add_argument("--ctx_ppl", type=int, default=1536)
    ap.add_argument("--bs_ppl", type=int, default=16)
    ap.add_argument("--q4", action="store_true")
    # Removed KL-related parameters
    args = ap.parse_args()

    # Load data
    print(f"Loading data from {args.fin}...")
    pool = []
    with open(args.fin, "r", encoding="utf-8") as f:
        for line in f:
            pool.append(json.loads(line))
            if len(pool) >= args.n: 
                break
    
    if not pool:
        raise SystemExit("No samples loaded from fastpass.")
    
    texts = [r["text"] for r in pool]
    print(f"[data] Loaded {len(texts)} samples")

    # Load models
    teacher_names = [s.strip() for s in args.teacher.split(",") if s.strip()]
    teachers, (S_tok, S) = load_models(teacher_names, args.student, q4=args.q4)

    # 计算Teacher PPL
    print("\n[Score] Computing teacher PPL...")
    ppl_T = ppl_models(texts, teachers, ctx=args.ctx_ppl, bs=args.bs_ppl)
    print(f"  Average PPL: {np.mean(ppl_T):.2f}, Median: {np.median(ppl_T):.2f}")

    # 计算Student PPL
    print("\n[Score] Computing student PPL...")
    ppl_S = ppl_models(texts, [(S_tok, S)], ctx=args.ctx_ppl, bs=args.bs_ppl)
    print(f"  Average PPL: {np.mean(ppl_S):.2f}, Median: {np.median(ppl_S):.2f}")

    # Calculate ΔNLL (replacing KL)
    nll_T = np.log(ppl_T + 1e-8)
    nll_S = np.log(ppl_S + 1e-8)
    delta = np.maximum(0.0, nll_S - nll_T)  # Part where student is worse than teacher
    
    print(f"\n[Score] ΔNLL stats:")
    print(f"  Average: {np.mean(delta):.4f}, Median: {np.median(delta):.4f}")
    print(f"  Min/Max: {np.min(delta):.4f}/{np.max(delta):.4f}")

    # ==== Soft gating scoring ====
    alpha = 1.0      # ΔNLL weight
    gamma = 0.7      # Structural reward weight (slightly increased)
    
    ppl_arr = np.asarray(ppl_T, dtype=np.float32)
    delta_arr = np.asarray(delta, dtype=np.float32)
    
    # Dynamic thresholds: operations in log space are more stable
    ppl_ln = np.log(ppl_arr + 1e-8)
    center = np.percentile(ppl_ln, 85)  # 85th percentile as center
    scale = max(0.25, np.std(ppl_ln))   # Standard deviation as width
    
    # Soft gating: sigmoid-form quality weight
    quality = 1.0 / (1.0 + np.exp((ppl_ln - center) / scale))
    
    # Calculate final scores
    scores = []
    for i, rec in enumerate(pool):
        bonus = format_bonus(rec)
        s = alpha * delta_arr[i] * float(quality[i]) + gamma * float(bonus)
        scores.append(s)
    
    # Build scored list
    scored = []
    for i, rec in enumerate(pool):
        out = dict(rec)
        out.update({
            "ppl_T": float(ppl_arr[i]),
            "ppl_S": float(ppl_S[i]),
            "delta_nll": float(delta_arr[i]),
            "quality": float(quality[i]),
            "score": float(scores[i]),
        })
        scored.append(out)
    
    print(f"\n[Score] Soft-gated score stats:")
    print(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    print(f"  Min/Max: {np.min(scores):.4f}/{np.max(scores):.4f}")
    print(f"  Quality weights: Mean={np.mean(quality):.3f}, Min/Max={np.min(quality):.3f}/{np.max(quality):.3f}")

    # ==== Train selector (ranking split ensures two classes) ====
    X, y = [], []
    for r in scored:
        X.append([
            len(r["text"]) / 1000,
            r["token_count"] / 100,
            int("function_calling" in r["tags"]),
            int("reasoning" in r["tags"]),
            int("roleplay" in r["tags"]),
            int("chatrag" in r["tags"]),
            np.log(r["ppl_T"] + 1) / 10,  # log scale for PPL
            int(r.get("json_ok", False)),
            r.get("json_depth", 0) / 5,
            r.get("json_keys", 0) / 10,
            r.get("steps", 0) / 10,
            r.get("turns", 0) / 10,
            int(r.get("fc_sig", False)),
            r.get("fc_params", 0) / 5,
            int(r.get("has_cite", False)),
        ])
        y.append(r["score"])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Ranking split function (ensures two classes)
    def make_rank_labels(y, pct=40):
        k = max(1, int(len(y) * pct / 100))
        idx = np.argsort(y)
        yb = np.zeros_like(y, dtype=np.int32)
        yb[idx[-k:]] = 1
        # If still single class, adjust to 30%
        if yb.sum() == 0 or yb.sum() == len(y):
            k = max(1, int(len(y) * 0.30))
            yb[:] = 0
            yb[idx[-k:]] = 1
        return yb
    
    y_bin = make_rank_labels(y, pct=40)
    print(f"\n[Score] Selector labels: {y_bin.sum()} positive / {len(y_bin)} total ({y_bin.mean():.1%})")
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    ])
    clf.fit(X, y_bin)

    # Save results
    with open(args.fout, "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    joblib.dump(clf, "selector.joblib")

    print("\n" + "="*50)
    print("Scoring Statistics Summary:")
    print(f"  Samples: {len(scored)}")
    print(f"  Teacher PPL: mean={np.mean(ppl_T):.2f}, median={np.median(ppl_T):.2f}")
    print(f"  Student PPL: mean={np.mean(ppl_S):.2f}, median={np.median(ppl_S):.2f}")
    print(f"  ΔNLL: mean={np.mean(delta):.4f}, median={np.median(delta):.4f}")
    print(f"  Score: mean={np.mean(y):.4f}, median={np.median(y):.4f}")
    print(f"  Positive ratio: {y_bin.mean():.2%}")
    print(f"✅ Saved: {args.fout} and selector.joblib")

if __name__ == "__main__":
    main()