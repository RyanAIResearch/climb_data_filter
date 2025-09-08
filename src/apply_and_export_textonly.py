"""
Full selector application + 3-phase difficulty-stratified mixed sampling + text_only format export
Enhanced version with multi-label bucketing, deduplication, and adaptive sampling
"""
import os
import json
import argparse
import numpy as np
import joblib
from collections import Counter
from tqdm import tqdm
import random
import re
from utils_simhash import simhash64

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def feat_row(r):
    """Extract feature vector (consistent with training)"""
    return [
        len(r["text"]) / 1000,
        r["token_count"] / 100,
        int("function_calling" in r["tags"]),
        int("reasoning" in r["tags"]),
        int("roleplay" in r["tags"]),
        int("chatrag" in r["tags"]),
        r.get("ppl_T", 10.0) / 10,
        int(r.get("json_ok", False)),
        r.get("json_depth", 0) / 5,
        r.get("json_keys", 0) / 10,
        r.get("steps", 0) / 10,
        r.get("turns", 0) / 10,
        int(r.get("fc_sig", False)),
        r.get("fc_params", 0) / 5,
        int(r.get("has_cite", False)),
    ]

def difficulty_bin(r):
    """
    Estimate difficulty based on task features
    Returns: 0(easy), 1(medium), 2(hard)
    """
    score = 0.0
    
    # Function-calling difficulty
    if "function_calling" in r["tags"]:
        score += 1.2
        if r.get("json_ok"):
            score += 0.6
        score += 0.3 * min(6, r.get("json_depth", 0))
        score += 0.2 * min(12, r.get("json_keys", 0))
        score += 0.1 * min(10, r.get("fc_params", 0))
    
    # Reasoning difficulty
    if "reasoning" in r["tags"]:
        score += 1.0
        score += 0.3 * min(12, r.get("steps", 0))
    
    # Roleplay difficulty
    if "roleplay" in r["tags"]:
        score += 0.9
        score += 0.2 * min(12, r.get("turns", 0))
    
    # ChatRAG difficulty
    if "chatrag" in r["tags"]:
        score += 0.9
        if r.get("has_cite"):
            score += 0.5
    
    # Length factors
    if r["token_count"] >= 512:
        score += 0.5
    if r["token_count"] >= 1024:
        score += 0.5
    
    # Classification (adjusted thresholds)
    if score >= 3.5:
        return 2  # hard
    elif score >= 1.8:
        return 1  # medium
    else:
        return 0  # easy

def mark_like_samples(rows):
    """Mark similar samples for backfill"""
    for r in rows:
        txt = r["text"]
        
        # FC-like marking
        r["fc_like"] = ("function_calling" in r["tags"]) or \
            bool(re.search(r'(?im)^\s*(Action|Action\s*Input|Thought|Observation)\s*:', txt)) or \
            bool(re.search(r'\b(GET|POST|PUT|DELETE|PATCH)\s+https?://', txt)) or \
            bool(re.search(r'"(tool_calls?|function_call|arguments?|params?)"\s*:', txt))
        
        # RAG-like marking
        r["rag_like"] = ("chatrag" in r["tags"]) or \
            ("http" in txt.lower()) or \
            bool(re.search(r'(?i)\b(ref(?:erences?)?|source|citation|bibliography)\b', txt)) or \
            bool(re.search(r'\[\d+\]', txt))

def write_textonly(texts, shard_id, out_dir, phase):
    """Write text_only format JSON file"""
    os.makedirs(out_dir, exist_ok=True)
    obj = {
        "type": "text_only",
        "instances": [{"text": t} for t in texts]
    }
    fname = os.path.join(out_dir, f"train_{phase}_{shard_id:05d}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def sample_bucket_by_difficulty(sorted_rows, token_cap, shard_cb, 
                               weights=(0.2, 0.4, 0.4), 
                               dup_policy=True, 
                               force_fc_dup=False,
                               seen=None):
    """
    Difficulty-stratified sampling with deduplication and FC oversampling support
    """
    if seen is None:
        seen = set()
    
    # Create difficulty bins
    bins = {0: [], 1: [], 2: []}
    for r in sorted_rows:
        b = difficulty_bin(r)
        bins[b].append(r)
    
    # Sort within each bin by sel_prob
    for b in bins:
        bins[b].sort(key=lambda x: x.get("sel_prob", 0.5), reverse=True)
    
    acc = 0
    buf = []
    shard_size = 10000
    ptr = {0: 0, 1: 0, 2: 0}
    local_used = 0
    
    # Sampling loop
    while acc < token_cap:
        made_progress = False
        
        for b, w in enumerate(weights):
            if w <= 0:
                continue
            
            # Number of samples to take from this difficulty per round
            take = max(1, int(64 * w))
            
            for _ in range(take):
                if ptr[b] >= len(bins[b]):
                    continue
                
                r = bins[b][ptr[b]]
                ptr[b] += 1
                
                # Deduplication check
                sid = r.get("simhash", simhash64(r["text"]))
                if sid in seen:
                    continue
                
                made_progress = True
                seen.add(sid)
                
                # Duplication strategy
                dup = 1
                if dup_policy:
                    prob = r.get("sel_prob", 0.5)
                    
                    # FC oversampling
                    if force_fc_dup and ("function_calling" in r["tags"] or r.get("fc_like")):
                        dup = 6 if prob < 0.9 else 10
                    else:
                        if prob > 0.9:
                            dup = 3
                        elif prob > 0.8:
                            dup = 2
                
                # Add to buffer (limit max duplicates per sample)
                for d in range(min(dup, 10)):
                    buf.append(r["text"])
                    acc += r["token_count"]
                    local_used += r["token_count"]
                    
                    if len(buf) >= shard_size:
                        shard_cb(buf)
                        buf = []
                    
                    if acc >= token_cap:
                        break
                
                if acc >= token_cap:
                    break
            
            if acc >= token_cap:
                break
        
        # If all bins are exhausted
        if not made_progress:
            break
    
    # Handle remaining buffer
    if buf:
        shard_cb(buf)
    
    return local_used

def refill_like(rows, like_key, need_tokens, shard_cb, weights, seen):
    """Backfill using similar pool"""
    like_pool = [r for r in rows 
                 if r.get(like_key) and r.get("sel_prob", 0) > 0.35
                 and r.get("simhash", simhash64(r["text"])) not in seen]
    
    if not like_pool:
        return 0
    
    # Sort by probability
    like_pool.sort(key=lambda x: x.get("sel_prob", 0.5), reverse=True)
    
    return sample_bucket_by_difficulty(
        like_pool, need_tokens, shard_cb, 
        weights=weights, dup_policy=True, seen=seen
    )

def boost_ratio(base, **boost):
    """Adjust ratios by boost coefficients"""
    r = dict(base)
    for k, v in boost.items():
        if k in r:
            r[k] = r[k] * v
    s = sum(r.values())
    if s <= 0:
        return base
    return {k: v/s for k, v in r.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastpass", required=True)
    ap.add_argument("--selector", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--token_budget", type=int, default=5_000_000_000)
    ap.add_argument("--early_frac", type=float, default=0.30)
    ap.add_argument("--mid_frac", type=float, default=0.40)
    ap.add_argument("--early_boost_func", type=float, default=2.5)   # Increased
    ap.add_argument("--early_boost_reason", type=float, default=1.6)
    ap.add_argument("--mid_boost_role", type=float, default=1.5)     # Increased
    ap.add_argument("--mid_boost_rag", type=float, default=1.4)      # Increased
    ap.add_argument("--val_frac", type=float, default=0.05)
    args = ap.parse_args()
    
    print(f"Loading selector from {args.selector}...")
    sel = joblib.load(args.selector)
    
    print(f"Loading data from {args.fastpass}...")
    rows = []
    with open(args.fastpass, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    
    # Mark similar samples
    print("Marking like samples for backfill...")
    mark_like_samples(rows)
    
    # Apply selector scoring
    print("Applying selector...")
    X = np.array([feat_row(r) for r in rows], dtype=np.float32)
    probs = sel.predict_proba(X)[:, 1]
    
    for i, r in enumerate(rows):
        r["sel_prob"] = float(probs[i])
    
    # Multi-label bucketing (allow same sample in multiple buckets)
    buckets = {
        "general": [],
        "function_calling": [],
        "reasoning": [],
        "roleplay": [],
        "chatrag": []
    }
    
    print("Multi-label bucketing...")
    for r in rows:
        has_tag = False
        
        # Add to all relevant buckets
        if "function_calling" in r["tags"]:
            buckets["function_calling"].append(r)
            has_tag = True
        if "reasoning" in r["tags"]:
            buckets["reasoning"].append(r)
            has_tag = True
        if "roleplay" in r["tags"]:
            buckets["roleplay"].append(r)
            has_tag = True
        if "chatrag" in r["tags"]:
            buckets["chatrag"].append(r)
            has_tag = True
        
        # If no special tags, add to general
        if not has_tag:
            buckets["general"].append(r)
    
    # Sort within each bucket by probability
    for k in buckets:
        buckets[k].sort(key=lambda x: x["sel_prob"], reverse=True)
    
    print("\nBucket sizes (multi-label):")
    for k, v in buckets.items():
        print(f"  {k}: {len(v):,}")
    
    # FC-like and RAG-like statistics
    fc_like_count = sum(1 for r in rows if r.get("fc_like"))
    rag_like_count = sum(1 for r in rows if r.get("rag_like"))
    print(f"\nLike pools: fc_like={fc_like_count:,}, rag_like={rag_like_count:,}")
    
    # Export validation set
    val_texts = []
    if args.val_frac > 0:
        print(f"\nExtracting {args.val_frac:.1%} validation set...")
        for bucket_name, bucket_rows in buckets.items():
            n_val = int(len(bucket_rows) * args.val_frac)
            if n_val > 0:
                # Sample from medium difficulty
                medium_rows = [r for r in bucket_rows if difficulty_bin(r) == 1]
                if medium_rows:
                    val_sample = random.sample(medium_rows, min(n_val, len(medium_rows)))
                    val_texts.extend([r["text"] for r in val_sample])
        
        if val_texts:
            val_obj = {
                "type": "text_only",
                "instances": [{"text": t} for t in val_texts[:10000]]  # cap at 10k
            }
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "val.json"), "w") as f:
                json.dump(val_obj, f, ensure_ascii=False)
            print(f"  Saved {len(val_obj['instances'])} validation samples")
    
    # Ratio settings (maintain FC/RAG proportions)
    late_ratio = {
        "general": 0.40,
        "function_calling": 0.18,  # Keep high
        "reasoning": 0.17,
        "roleplay": 0.12,           # Increase
        "chatrag": 0.13             # Increase
    }
    
    # Apply boost
    early_ratio = boost_ratio(
        late_ratio,
        function_calling=args.early_boost_func,
        reasoning=args.early_boost_reason
    )
    mid_ratio = boost_ratio(
        late_ratio,
        roleplay=args.mid_boost_role,
        chatrag=args.mid_boost_rag
    )
    
    # Calculate token budget for each phase
    total = args.token_budget
    early = int(total * args.early_frac)
    mid = int(total * args.mid_frac)
    late = total - early - mid
    
    # Difficulty weights (more balanced)
    EARLY_WEIGHTS = (0.20, 0.40, 0.40)  # Balanced but still hard-leaning
    MID_WEIGHTS = (0.25, 0.50, 0.25)    # Focus on medium
    LATE_WEIGHTS = (0.35, 0.45, 0.20)   # More easy samples
    
    # Sampling execution
    print(f"\n=== Starting 3-phase sampling ===")
    print(f"Total budget: {total:,} tokens")
    print(f"Early: {early:,} | Mid: {mid:,} | Late: {late:,}")
    
    used = 0
    per_bucket = Counter()
    per_phase = Counter()
    shard_id = 0
    seen_global = set()  # Global deduplication
    
    def make_callback(phase):
        def cb(buf):
            nonlocal shard_id
            write_textonly(buf, shard_id, args.out_dir, phase)
            shard_id += 1
        return cb
    
    # Three-phase sampling
    for phase_name, ratio, cap, weights in [
        ("early", early_ratio, early, EARLY_WEIGHTS),
        ("mid", mid_ratio, mid, MID_WEIGHTS),
        ("late", late_ratio, late, LATE_WEIGHTS),
    ]:
        print(f"\n{phase_name.upper()} phase:")
        phase_used = 0
        phase_targets = {k: int(cap * v) for k, v in ratio.items()}
        
        for bucket_name, bucket_ratio in ratio.items():
            token_cap = phase_targets[bucket_name]
            if token_cap <= 0:
                continue
            
            # FC bucket uses forced oversampling
            force_fc = (bucket_name == "function_calling")
            
            got = sample_bucket_by_difficulty(
                buckets[bucket_name],
                token_cap,
                make_callback(phase_name),
                weights=weights,
                dup_policy=True,
                force_fc_dup=force_fc,
                seen=seen_global
            )
            
            used += got
            phase_used += got
            per_bucket[bucket_name] += got
            
            print(f"  {bucket_name}: {got:,} tokens (target: {token_cap:,})")
            
            # If severely insufficient, use similar backfill
            if got < token_cap * 0.7:  # Less than 70% of target
                need = token_cap - got
                print(f"    Backfilling {bucket_name} with {need:,} tokens...")
                
                if bucket_name == "function_calling":
                    extra = refill_like(rows, "fc_like", need, 
                                      make_callback(phase_name), 
                                      weights, seen_global)
                elif bucket_name == "chatrag":
                    extra = refill_like(rows, "rag_like", need,
                                      make_callback(phase_name),
                                      weights, seen_global)
                else:
                    extra = 0
                
                if extra > 0:
                    used += extra
                    phase_used += extra
                    per_bucket[bucket_name] += extra
                    print(f"    Added {extra:,} tokens from like pool")
        
        per_phase[phase_name] = phase_used
        
        # If this phase is not fully used, supplement with general
        if phase_used < cap * 0.95:  # Allow 5% under-utilization
            need = cap - phase_used
            print(f"  Filling remaining {need:,} tokens with general...")
            got = sample_bucket_by_difficulty(
                buckets["general"],
                need,
                make_callback(phase_name),
                weights=weights,
                dup_policy=False,
                seen=seen_global
            )
            used += got
            per_bucket["general"] += got
            per_phase[phase_name] += got
    
    # Generate report
    utilization = used / total if total > 0 else 0
    report = {
        "token_budget": total,
        "used_tokens": used,
        "utilization": f"{utilization:.1%}",
        "bucket_tokens": dict(per_bucket),
        "bucket_percentages": {k: f"{v/used:.1%}" for k, v in per_bucket.items()} if used > 0 else {},
        "phase_tokens": dict(per_phase),
        "ratios": {
            "early": {k: f"{v:.1%}" for k, v in early_ratio.items()},
            "mid": {k: f"{v:.1%}" for k, v in mid_ratio.items()},
            "late": {k: f"{v:.1%}" for k, v in late_ratio.items()},
        },
        "phase_fractions": {
            "early": args.early_frac,
            "mid": args.mid_frac,
            "late": 1 - args.early_frac - args.mid_frac,
        },
        "boosts": {
            "early_func": args.early_boost_func,
            "early_reason": args.early_boost_reason,
            "mid_role": args.mid_boost_role,
            "mid_rag": args.mid_boost_rag,
        },
        "unique_samples_used": len(seen_global),
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "budget_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== Export Complete ===")
    print(f"Total used: {used:,} / {total:,} tokens ({utilization:.1%})")
    print(f"Unique samples: {len(seen_global):,}")
    print(f"Output dir: {args.out_dir}")
    print(f"Report saved: budget_report.json")
    
    # Warn if utilization is low
    if utilization < 0.8:
        print(f"\n⚠️  WARNING: Budget utilization is only {utilization:.1%}")
        print("  Consider increasing MAX_ROWS or adjusting sampling parameters")

if __name__ == "__main__":
    main()