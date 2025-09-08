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
