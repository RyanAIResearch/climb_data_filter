# 

#!/usr/bin/env bash
set -euo pipefail

# ===== Adjustable Parameters (v3.0 - Optimized Version) =====
TEACHER="Qwen/Qwen2-7B"
STUDENT="data4elm/Llama-400M-12L"
TOKEN_BUDGET=3000000000  # Start with 3B, can increase to 4B if needed

# Expand candidate pool to ensure sufficient supply
MAX_ROWS=2000000        # 2M samples for better coverage
SMALL_N=60000           # 60k for scoring
SIMHASH_HAM=6           # Slightly relaxed deduplication

# Three-stage curriculum
EARLY_FRAC=0.30
MID_FRAC=0.40

# Stage weighting - enhance function and reasoning
EARLY_BOOST_FUNC=2.5    # Increased for FC
EARLY_BOOST_REASON=1.6  
MID_BOOST_ROLE=1.5      # Increased for roleplay
MID_BOOST_RAG=1.4       # Increased for RAG

# File paths
FASTPASS_OUT="fastpass.jsonl"
SCORED_SMALL="scored_small.jsonl"
SELECTOR="selector.joblib"
EXPORT_DIR="data/filtered_textonly"

mkdir -p "$EXPORT_DIR" src

# Copy the utils_simhash.py if it doesn't exist
if [ ! -f "utils_simhash.py" ]; then
    cat > utils_simhash.py << 'EOF'
"""
SimHash tool: 64-bit fingerprint + Hamming distance deduplication
"""
import re
import hashlib

def _tokens_for_simhash(text):
    """Extract 3-grams as tokens for simhash"""
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < 3:
        return [s]
    return [s[i:i+3] for i in range(len(s)-2)]

def simhash64(text):
    """Calculate 64-bit SimHash fingerprint"""
    bits = [0] * 64
    tokens = _tokens_for_simhash(text)
    
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for b in range(64):
            if (h >> b) & 1:
                bits[b] += 1
            else:
                bits[b] -= 1
    
    v = 0
    for b in range(64):
        if bits[b] > 0:
            v |= (1 << b)
    return v

def hamming64(a, b):
    """Calculate Hamming distance between two 64-bit integers"""
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")
EOF
fi

echo "=== [A] Streaming rough filtering + bucketing + structural features (Enhanced Detection) ==="
python src/climblab_fastpass.py \
  --out "$FASTPASS_OUT" \
  --max_rows "$MAX_ROWS" \
  --simhash_ham "$SIMHASH_HAM"

# Quick check of bucket distribution
echo -e "\n=== Checking bucket distribution ==="
python -c "
import json
from collections import Counter
buckets = Counter()
with open('$FASTPASS_OUT') as f:
    for line in f:
        r = json.loads(line)
        for tag in r.get('tags', []):
            buckets[tag] += 1
print('Bucket counts:')
for k, v in buckets.most_common():
    print(f'  {k}: {v:,}')
"

echo -e "\n=== [B] Small sample teacher-student scoring (ΔNLL) + training lightweight selector ==="
python src/score_small_and_train_selector.py \
  --in "$FASTPASS_OUT" \
  --out "$SCORED_SMALL" \
  --n "$SMALL_N" \
  --teacher "$TEACHER" \
  --student "$STUDENT" \
  --ctx_ppl 1536 \
  --bs_ppl 24 \
  --q4

echo -e "\n=== [C] Full selection → three-stage (difficulty stratified) mixed sampling → export text_only ==="
python src/apply_and_export_textonly.py \
  --fastpass "$FASTPASS_OUT" \
  --selector "$SELECTOR" \
  --out_dir "$EXPORT_DIR" \
  --token_budget "$TOKEN_BUDGET" \
  --early_frac "$EARLY_FRAC" \
  --mid_frac "$MID_FRAC" \
  --early_boost_func "$EARLY_BOOST_FUNC" \
  --early_boost_reason "$EARLY_BOOST_REASON" \
  --mid_boost_role "$MID_BOOST_ROLE" \
  --mid_boost_rag "$MID_BOOST_RAG" \
  --val_frac 0.05

echo -e "\n=== Checking export results ==="
echo "Files in $EXPORT_DIR:"
ls -lh "$EXPORT_DIR"/*.json | head -20

echo -e "\nBudget report:"
if [ -f "$EXPORT_DIR/budget_report.json" ]; then
    python -c "
import json
with open('$EXPORT_DIR/budget_report.json') as f:
    report = json.load(f)
print(f'Utilization: {report[\"utilization\"]}')
print(f'Total used: {report[\"used_tokens\"]:,} tokens')
print(f'Unique samples: {report.get(\"unique_samples_used\", \"N/A\")}')
print('\\nBucket distribution:')
for k, v in report.get('bucket_percentages', {}).items():
    print(f'  {k}: {v}')
"
fi

echo -e "\n✅ Data export completed: $EXPORT_DIR"
echo "➡ Next step: bash train_and_eval.sh for DoRA single-round training, merging and ELMB evaluation"

# Warning if low utilization
python -c "
import json
with open('$EXPORT_DIR/budget_report.json') as f:
    report = json.load(f)
util = float(report['utilization'].rstrip('%')) / 100
if util < 0.8:
    print(f'\\n⚠️  WARNING: Budget utilization is only {util:.1%}')
    print('  Consider increasing MAX_ROWS or adjusting parameters')
"