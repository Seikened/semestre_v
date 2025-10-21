# fast_tokenizer_numba.py
# Tokenizador minimalista y rápido sobre uint8 + Numba.
# Extras:
#  - Métricas por fase: escaneo puro vs construcción de strings
#  - Ruta "IDs sin strings" (hash por token en el kernel) para evitar objetos Python
#  - Baselines: split() y regex
#  - API separada: tokens (strings) / ids (hash-based) / fases

import os, time, re
import numpy as np
from numba import njit, uint64

# ============================ CONFIG GLOBAL ===================================
# Hilos de Numba (el escáner es secuencial; esto es opcional)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# Cambia reglas aquí si quieres otro set de separadores.
SPLIT_ON_WHITESPACE = True   # si True, añade whitespace ASCII como separador
SPLIT_ON_PUNCT      = True   # si True, separa en puntuación ASCII básica

# ============================ TABLA DE CLASES =================================
def build_sep_table(split_ws=True, split_punct=True) -> np.ndarray:
    sep = np.zeros(256, dtype=np.uint8)
    if split_ws:
        for b in (9, 10, 11, 12, 13, 32):  # \t \n \v \f \r ' '
            sep[b] = 1
    if split_punct:
        punct = b"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
        for b in punct:
            sep[int(b)] = 1
    return sep

SEP = build_sep_table(SPLIT_ON_WHITESPACE, SPLIT_ON_PUNCT)

# =============================== KERNELS ======================================
@njit(cache=True, fastmath=False)
def scan_tokens(u8, sep, starts, ends):
    """
    Escaneo en un solo pase: emite offsets [start,end) por token.
    Sin temporales ni objetos Python.
    """
    n = u8.size
    i = 0
    tcount = 0
    # saltar separadores iniciales
    while i < n and sep[u8[i]] == 1:
        i += 1
    while i < n:
        s = i
        i += 1
        while i < n and sep[u8[i]] == 0:
            i += 1
        e = i
        starts[tcount] = s
        ends[tcount] = e
        tcount += 1
        while i < n and sep[u8[i]] == 1:
            i += 1
    return tcount

@njit(cache=True, fastmath=False)
def scan_tokens_hash(u8, sep, starts, ends, hashes):
    """
    Igual que scan_tokens, pero **calcula hash FNV-1a** por token dentro del bucle.
    Evita crear strings para IDs; ideal para pipeline numérico.
    """
    FNV_OFFSET = uint64(1469598103934665603)   # 64-bit
    FNV_PRIME  = uint64(1099511628211)

    n = u8.size
    i = 0
    tcount = 0
    while i < n and sep[u8[i]] == 1:
        i += 1
    while i < n:
        s = i
        h = FNV_OFFSET
        # consumir primer byte del token
        b = u8[i]
        h ^= uint64(b)
        h *= FNV_PRIME
        i += 1
        # resto del token
        while i < n and sep[u8[i]] == 0:
            b = u8[i]
            h ^= uint64(b)
            h *= FNV_PRIME
            i += 1
        e = i
        starts[tcount] = s
        ends[tcount]   = e
        hashes[tcount] = h
        tcount += 1
        while i < n and sep[u8[i]] == 1:
            i += 1
    return tcount

# =============================== WRAPPERS =====================================
def tokenize_numba_offsets(text: str):
    """
    Devuelve sólo offsets (y el buffer u8) para usos downstream sin strings.
    """
    u8 = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    starts = np.empty(u8.size, dtype=np.int32)
    ends   = np.empty(u8.size, dtype=np.int32)
    count  = scan_tokens(u8, SEP, starts, ends)
    return u8, starts[:count], ends[:count]

def tokenize_numba_tokens(text: str):
    """
    Devuelve lista de tokens como strings (conversión fuera del hot loop).
    """
    u8, S, E = tokenize_numba_offsets(text)
    # construcción de strings: costo dominante si hay muchos tokens
    tokens = [u8[s:e].tobytes().decode("utf-8", errors="ignore") for s, e in zip(S, E)]
    return tokens

def tokenize_numba_ids(text: str, vocab=None, build_vocab=True, unk_token="<unk>"):
    """
    Ruta de **IDs sin strings**: el kernel emite hashes por token.
    Luego mapeamos hash->id en Python (sin convertir a str).
    Nota: hashes pueden colisionar; para demo/bench es suficiente.
    """
    u8 = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    starts = np.empty(u8.size, dtype=np.int32)
    ends   = np.empty(u8.size, dtype=np.int32)
    hashes = np.empty(u8.size, dtype=np.uint64)
    count  = scan_tokens_hash(u8, SEP, starts, ends, hashes)

    H = hashes[:count]
    # construir/usar vocabulario basado en hash
    if vocab is None and build_vocab:
        vocab = {}  # hash -> id
        ids = np.empty(count, dtype=np.int32)
        next_id = 0
        for i in range(count):
            h = int(H[i])
            vid = vocab.get(h)
            if vid is None:
                vid = next_id
                vocab[h] = vid
                next_id += 1
            ids[i] = vid
        return ids, vocab
    else:
        if vocab is None:
            # sin vocab: regresa los hashes (IDs provisionales)
            return H.copy(), {}
        unk_id = vocab.get(unk_token, -1) if isinstance(unk_token, str) else -1
        ids = np.array([vocab.get(int(h), unk_id) for h in H], dtype=np.int32)
        return ids, vocab

# =============================== BASELINES ====================================
def tokenize_python_ws(text: str):
    # Fast-path C para whitespace
    return [t for t in text.split() if t]

def tokenize_python_regex(text: str):
    # Similar semántica a "whitespace+punct", pero con costo de regex
    return [t for t in re.split(r"[\s\W]+", text) if t]

# ============================== BENCHMARKS ====================================
def bench_all():
    para = ("Numba is fast. Tokenizers love contiguous bytes, "
            "single-pass scans, and no Python objects in the hot loop! ") * 2000
    text = para * 16  # ~pocos MB; sube si quieres

    # Warm-up kernels
    _ = tokenize_numba_offsets(text)
    _ = tokenize_numba_ids(text, build_vocab=False)

    # --- Baselines Python ---
    t0 = time.perf_counter(); tok_ws = tokenize_python_ws(text); t1 = time.perf_counter()
    t_py = t1 - t0

    t0 = time.perf_counter(); tok_rgx = tokenize_python_regex(text); t1 = time.perf_counter()
    t_re = t1 - t0

    # --- Numba: tokens (incluye strings) ---
    t0 = time.perf_counter(); tok_nb = tokenize_numba_tokens(text); t1 = time.perf_counter()
    t_nb_tokens = t1 - t0

    # --- Numba: fases separadas ---
    # Escaneo puro
    u8 = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    starts = np.empty(u8.size, dtype=np.int32)
    ends   = np.empty(u8.size, dtype=np.int32)
    _ = scan_tokens(u8, SEP, starts, ends)  # warm
    t0 = time.perf_counter()
    count = scan_tokens(u8, SEP, starts, ends)
    t1 = time.perf_counter()
    scan_time = t1 - t0

    # Construcción de strings (fuera del kernel)
    S = starts[:count]; E = ends[:count]
    t2 = time.perf_counter()
    toks = [u8[s:e].tobytes().decode("utf-8", errors="ignore") for s, e in zip(S, E)]
    t3 = time.perf_counter()
    build_time = t3 - t2

    # --- Numba: IDs sin strings (hash) ---
    starts[:] = 0; ends[:] = 0
    hashes = np.empty(u8.size, dtype=np.uint64)
    _ = scan_tokens_hash(u8, SEP, starts, ends, hashes)  # warm
    t0 = time.perf_counter()
    count_h = scan_tokens_hash(u8, SEP, starts, ends, hashes)
    t1 = time.perf_counter()
    scan_hash_time = t1 - t0

    # mapear hash->id sin strings (vocab dinámico)
    H = hashes[:count_h]
    t2 = time.perf_counter()
    vocab = {}
    ids = np.empty(count_h, dtype=np.int32)
    next_id = 0
    for i in range(count_h):
        h = int(H[i])
        vid = vocab.get(h)
        if vid is None:
            vid = next_id
            vocab[h] = vid
            next_id += 1
        ids[i] = vid
    t3 = time.perf_counter()
    ids_map_time = t3 - t2

    # --- Resultados ---
    print(f"Python .split           : {t_py:.3f} s  | tokens={len(tok_ws)}")
    print(f"Python regex            : {t_re:.3f} s  | tokens={len(tok_rgx)}")
    print(f"Numba tokens (1 pase)   : {t_nb_tokens:.3f} s  | tokens={len(tok_nb)}")
    print(f"Numba phases -> scan     {scan_time:.3f} s  | build_str {build_time:.3f} s | tokens={len(toks)}")
    print(f"Numba IDs   -> scan+hash {scan_hash_time:.3f} s  | id_map   {ids_map_time:.3f} s | ids={ids.size}")
    # Nota: típicamente verás scan << build_str; y scan_hash similar a scan,
    # mientras id_map es mucho más barato que construir strings.

# ================================ DEMO ========================================
if __name__ == "__main__":
    import time
    bench_all()

    demo = "Hola Fer! Numba+uint8, un solo pase; luego IDs sin strings. ¿Listo?"
    print("\n--- Demo tokens ---")
    print(tokenize_numba_tokens(demo))
    print("--- Demo ids (hash) ---")
    ids, vocab = tokenize_numba_ids(demo, build_vocab=True)
    print("ids[:10]:", ids[:10], "| vocab_size:", len(vocab))