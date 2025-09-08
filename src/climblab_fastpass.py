"""
Stream processing ClimbLab data: decode tokens, coarse filtering, bucketing, extract structural features
Enhanced version with better function_calling and chatrag detection
"""
import re
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import tiktoken
from utils_simhash import simhash64, hamming64
import random
import numpy as np

# GPT-2 tokenizer for decoding
enc = tiktoken.get_encoding("gpt2")
TOKEN_KEYS = ["tokens", "input_ids", "token_ids", "token_ids_gpt2"]

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def get_tokens(row):
    """Extract token list from data row"""
    for k in TOKEN_KEYS:
        if k in row and isinstance(row[k], (list, tuple)):
            return row[k]
    return None

def ok_len(n):
    """Length filtering: 16-2048 tokens (relaxed from 32)"""
    return 16 <= n <= 2048  # Relaxed minimum from 32 to 16

def ascii_ratio(s):
    """ASCII character ratio (filter non-English)"""
    if not s:
        return 0
    return sum(c.isascii() for c in s) / len(s)

def bucket_and_features(text: str):
    """
    Bucket text and extract features (enhanced version - expanded function-calling and chatrag recognition)
    Returns: (tags list, features dict)
    """
    tags = []
    feats = {
        "json_ok": False,
        "json_depth": 0,
        "json_keys": 0,
        "steps": 0,
        "has_cite": False,
        "turns": 0,
        "has_roles": False,
        "fc_sig": False,
        "fc_params": 0
    }

    # --------- Function-calling: Extended recognition ---------
    def _json_relax(s: str) -> str:
        """Relaxed JSON cleaning: remove comments, add quotes, replace keywords, remove trailing commas"""
        # Remove comments
        s = re.sub(r'//.*?$', '', s, flags=re.M)
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
        # Add quotes to unquoted keys
        s = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', s)
        # Convert single quotes to double quotes
        s = s.replace("'", '"')
        # Convert Python keywords to JSON
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        s = re.sub(r'\bNone\b', 'null', s)
        # Remove trailing commas
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    json_match = None

    # Case A: ```json code blocks
    code_blocks = re.findall(r"```(?:json|JSON)?\s*([\s\S]{0,8192}?)```", text)
    if code_blocks:
        json_match = code_blocks[0]

    # Case B: Brace segments
    if not json_match:
        brace_match = re.search(r"\{[\s\S]{0,4096}\}", text)
        if brace_match:
            snippet = brace_match.group(0)
            # Extended API/tool call fields
            api_keywords = [
                '"arguments"', '"params"', '"parameters"', '"tool_calls"',
                '"name"', '"function"', '"schema"', '"inputs"', '"outputs"',
                '"method"', '"endpoint"', '"request"', '"response"',
                '"assistant_tool_calls"', '"tool_call"', '"tool_name"',
                '"properties"', '"required"', '"responses"', '"paths"',
                '"finish_reason":"tool_calls"'
            ]
            if any(k in snippet for k in api_keywords):
                json_match = snippet

    # Parse JSON
    if json_match:
        obj = None
        # First try strict parsing
        try:
            obj = json.loads(json_match)
        except:
            # If failed, try relaxed parsing
            try:
                relaxed = _json_relax(json_match)
                obj = json.loads(relaxed)
            except:
                obj = None

        if isinstance(obj, (dict, list)):
            feats["json_ok"] = True

            def depth(o, d=0):
                if isinstance(o, dict):
                    if not o: 
                        return d + 1
                    return max([d + 1] + [depth(v, d + 1) for v in o.values()])
                if isinstance(o, list):
                    if not o: 
                        return d + 1
                    return max([d + 1] + [depth(v, d + 1) for v in o])
                return d + 1

            feats["json_depth"] = min(10, depth(obj))
            if isinstance(obj, dict):
                feats["json_keys"] = min(20, len(obj.keys()))
            
            if "function_calling" not in tags:
                tags.append("function_calling")

    # ReAct style detection
    react_patterns = [r'(?im)^\s*(Thought|Action|Action\s*Input|Observation)\s*:']
    if any(re.search(p, text) for p in react_patterns):
        if "function_calling" not in tags:
            tags.append("function_calling")
        feats["fc_sig"] = True
        feats["fc_params"] = max(feats["fc_params"], 2)

    # HTTP/API call style
    http_patterns = [
        r'\b(GET|POST|PUT|DELETE|PATCH|HEAD)\s+https?://',
        r'curl\s+-X\s+(GET|POST|PUT|DELETE|PATCH)',
        r'"Content-Type"\s*:\s*"application/json"',
        r'Authorization:\s*Bearer\s+\S+'
    ]
    if any(re.search(p, text) for p in http_patterns):
        if "function_calling" not in tags:
            tags.append("function_calling")
        feats["fc_params"] = max(feats["fc_params"], 2)

    # Tool call patterns
    tool_patterns = [
        r'"tool_calls?"\s*:',
        r'"function_call"\s*:',
        r'<function=\w+>',
        r'<tool_call>',
        r'"assistant_tool_calls"\s*:',
        r'"tool_name"\s*:'
    ]
    for pattern in tool_patterns:
        if re.search(pattern, text):
            if "function_calling" not in tags:
                tags.append("function_calling")
            feats["fc_params"] = max(feats["fc_params"], 1)
            break

    # General function calls (not definitions)
    if re.search(r'\b[a-zA-Z_]\w*\s*\([^)\n]{1,200}\)', text) and not re.search(r'\b(def|function|class)\b', text):
        if "function_calling" not in tags:
            tags.append("function_calling")
        feats["fc_sig"] = True

    # Function signature detection (extended)
    function_patterns = [
        r'"function"\s*:\s*"[A-Za-z_]\w*"',
        r'\bdef\s+\w+\s*\(',
        r'\basync\s+function\b',
        r'function\s+\w+\s*\(',
        r'const\s+\w+\s*=\s*\([^)]*\)\s*=>'  # Arrow functions
    ]
    for pattern in function_patterns:
        if re.search(pattern, text):
            feats["fc_sig"] = True
            break

    # Parameter field counting
    param_keywords = [
        r'"(args?|parameters?|params?|tool_calls?|tools?|inputs?|arguments?)"\s*:',
    ]
    param_count = 0
    for pattern in param_keywords:
        param_count += len(re.findall(pattern, text))
    feats["fc_params"] = min(10, param_count)

    # --------- Roleplay: Enhanced roleplay detection ---------
    role_patterns = [
        r"^(System|User|Assistant|Human|Bot|AI|Agent)\s*:",
        r"^\*\*?(System|User|Assistant|Human|Bot|AI|Agent)\*\*?\s*:",
        r"^\*[^*]+\*",  # Action markers
        r"^\([^)]+\)",  # Stage directions
    ]
    role_lines = []
    for pattern in role_patterns:
        role_lines.extend(re.findall(pattern, text, flags=re.M))
    
    feats["turns"] = len(role_lines)
    feats["has_roles"] = len(role_lines) > 0
    
    # Extended roleplay trigger words
    roleplay_triggers = [
        r"(Act as|You are|You're playing|Roleplay as|Pretend to be|Imagine you are)",
        r"(In this scenario|In character as|Speaking as)",
        r"(in-character|out-of-character|OOC)",
        r"<(character|persona|role)>",
    ]
    has_roleplay_trigger = False
    for pattern in roleplay_triggers:
        if re.search(pattern, text, re.I):
            has_roleplay_trigger = True
            break
    
    if feats["turns"] >= 1 or has_roleplay_trigger:  # Lowered threshold
        tags.append("roleplay")

    # --------- Reasoning: Reasoning step detection ---------
    step_patterns = [
        r"(^|\s)(Step\s*\d+\.?)",
        r"(^|\n)(\d+[\.\)]\s+\w)",
        r"(First|Second|Third|Next|Finally|Initially|Subsequently),?\s",
    ]
    steps = 0
    for pattern in step_patterns:
        steps += len(re.findall(pattern, text, re.I))
    feats["steps"] = min(20, steps)
    
    # Reasoning keywords
    reasoning_keywords = [
        r"\b(Therefore|Hence|Proof|QED|Thus|Let us prove|It follows that|We conclude)",
        r"\b(Assume|Suppose|Given that|By definition|By theorem)",
        r"\b(This implies|This means|Consequently|As a result)",
    ]
    has_reasoning = False
    for pattern in reasoning_keywords:
        if re.search(pattern, text, re.I):
            has_reasoning = True
            break
    
    if steps > 0 or has_reasoning:
        tags.append("reasoning")

    # --------- ChatRAG: Extended citation and reference detection ---------
    citation_patterns = [
        r"https?://[\w\./\-#%&\?=]+",  # URLs
        r"\bdoi:\s*[\w\./\-]+",         # DOI
        r"\barXiv:\s*[\d\.]+",          # arXiv
        r"\b(References?|Citations?|Sources?|Bibliography):",
        r"\[\d+\]",                     # Numeric citations
        r"\([A-Z][a-z]+ et al\.?,? \d{4}\)",  # Academic citation format
        r"\([A-Z][a-z]+,? \d{4}\)",     # Simple author-year format
    ]
    
    # Extended RAG keywords
    rag_extra = [
        r'(?i)\b(source|ref(?:erences?)?|works\s+cited|citations?)\b',
        r'\[\w+\]\s*https?://',
        r'(?i)\b(accessed|retrieved)\s+on\b'
    ]
    
    for pattern in citation_patterns + rag_extra:
        if re.search(pattern, text, re.I):
            feats["has_cite"] = True
            if "chatrag" not in tags:
                tags.append("chatrag")
            break

    # Default classification
    if not tags:
        tags = ["general"]

    return tags, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=1000000)
    ap.add_argument("--simhash_ham", type=int, default=5)
    args = ap.parse_args()
    
    print(f"Loading ClimbLab dataset (streaming mode)...")
    ds = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    
    sim_seen = []
    kept = 0
    total_tokens = 0
    bucket_counts = {"general": 0, "function_calling": 0, "reasoning": 0, "roleplay": 0, "chatrag": 0}
    
    with open(args.out, "w", encoding="utf-8") as f:
        pbar = tqdm(desc="Processing", total=args.max_rows)
        
        for row in ds:
            toks = get_tokens(row)
            if toks is None or not ok_len(len(toks)):
                continue
            
            try:
                text = enc.decode(toks)
            except:
                continue
            
            # Language filtering
            if ascii_ratio(text) < 0.6:
                continue
            
            # Near-duplicate deduplication
            sig = simhash64(text)
            dup = False
            for s in sim_seen[-10000:]:  # Only compare with recent 10k to avoid O(n^2)
                if hamming64(sig, s) <= args.simhash_ham:
                    dup = True
                    break
            if dup:
                continue
            sim_seen.append(sig)
            
            # Bucketing and feature extraction
            tags, feats = bucket_and_features(text)
            
            rec = {
                "text": text,
                "token_count": len(toks),
                "tags": tags,
                "simhash": int(sig),  # Store for deduplication later
                **feats  # Expand all features
            }
            
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            total_tokens += len(toks)
            
            # Statistics
            for tag in tags:
                if tag in bucket_counts:
                    bucket_counts[tag] += 1
            
            pbar.update(1)
            if kept >= args.max_rows:
                break
        
        pbar.close()
    
    print(f"\n=== Fastpass Statistics ===")
    print(f"Total kept: {kept:,} samples")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Bucket distribution:")
    for k, v in bucket_counts.items():
        print(f"  {k}: {v:,}")

if __name__ == "__main__":
    main()