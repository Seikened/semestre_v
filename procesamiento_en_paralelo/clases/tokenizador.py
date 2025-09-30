# fast_tokenizer_numba.py
# A minimal high-throughput whitespace/punct tokenizer using Numba over uint8 buffers.
# Design goals: single pass over bytes, no Python objects in the hot loop, zero temporaries.

import os, time, re
import numpy as np
from numba import njit

# Optional: align Numba threads (not strictly needed; this scanner is sequential by design).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# ---------------------------
# 1) Build a byte-class table
# ---------------------------
# We'll mark "separator" bytes (ASCII whitespace + basic punctuation).
def build_sep_table():
    sep = np.zeros(256, dtype=np.uint8)
    # ASCII whitespace
    for b in (9, 10, 11, 12, 13, 32):  # \t \n \v \f \r ' '
        sep[b] = 1
    # Basic punctuation to split on (tune as you like)
    punct = b"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
    for b in punct:
        sep[int(b)] = 1
    return sep

SEP = build_sep_table()

# ----------------------------------------
# 2) Numba kernel: one-pass token scanning
# ----------------------------------------
# Input: u8 bytes and sep table
# Output: starts[], ends[], and count of tokens (write-once, no temps)
@njit(cache=True, fastmath=False)  # sequential on purpose; predictable branchy code
def scan_tokens(u8, sep, starts, ends):
    n = u8.size
    i = 0
    tcount = 0
    # Skip leading seps
    while i < n and sep[u8[i]] == 1:
        i += 1
    while i < n:
        # Start of token
        s = i
        i += 1
        # Advance while non-separator
        while i < n and sep[u8[i]] == 0:
            i += 1
        e = i
        # Emit token
        starts[tcount] = s
        ends[tcount] = e
        tcount += 1
        # Skip run of separators
        while i < n and sep[u8[i]] == 1:
            i += 1
    return tcount

# ---------------------------------------------------
# 3) Friendly Python wrapper: bytes -> tokens / ids
# ---------------------------------------------------
def tokenize_numba(text: str, return_ids=False, vocab=None):
    # Encode once; operate on contiguous bytes
    u8 = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    # Upper-bound on #tokens is len(bytes) (worst case: "a a a ...")
    starts = np.empty(u8.size, dtype=np.int32)
    ends   = np.empty(u8.size, dtype=np.int32)
    count = scan_tokens(u8, SEP, starts, ends)

    # Slice only the tokens found
    starts = starts[:count]
    ends   = ends[:count]

    # Turn offsets into strings (done out of the hot loop)
    tokens = [u8[s:e].tobytes().decode("utf-8", errors="ignore") for s, e in zip(starts, ends)]

    if not return_ids:
        return tokens

    # Optional: map tokens to ids. Keep it simple with a Python dict (outside hot loop).
    # If no vocab provided, build one on the fly.
    if vocab is None:
        vocab = {}
        next_id = 0
        ids = []
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
            ids.append(vocab[tok])
        return tokens, np.array(ids, dtype=np.int32), vocab
    else:
        unk_id = vocab.get("<unk>", -1)
        ids = np.array([vocab.get(tok, unk_id) for tok in tokens], dtype=np.int32)
        return tokens, ids, vocab

# ----------------------------------------------------
# 4) Baselines for sanity: pure Python & regex variant
# ----------------------------------------------------
def tokenize_python(text: str):
    # Simple split on whitespace and punctuation (roughly similar behavior)
    return [t for t in re.split(r"[\s\W]+", text) if t]

def tokenize_python_ws(text: str):
    # Just whitespace, closest to many .split() baselines
    return [t for t in text.split() if t]

# ------------------------
# 5) Tiny benchmark helper
# ------------------------
def bench():
    # Make a decently sized text by repeating a paragraph
    para = ("Numba is fast. Tokenizers love contiguous bytes, "
            "single-pass scans, and no Python objects in the hot loop! ") * 2000
    text = para * 16  # ~ a few MB; adjust up for your machine

    # Warm up
    _ = tokenize_numba(text)

    t0 = time.perf_counter()
    tok0 = tokenize_python_ws(text)
    t1 = time.perf_counter()
    t_py = t1 - t0

    t0 = time.perf_counter()
    tok1 = tokenize_python(text)
    t1 = time.perf_counter()
    t_re = t1 - t0

    t0 = time.perf_counter()
    tok2 = tokenize_numba(text)
    t1 = time.perf_counter()
    t_nb = t1 - t0

    # Sanity: token counts roughly similar (regex may differ on rules)
    print(f"Python .split   : {t_py:.3f} s  | tokens={len(tok0)}")
    print(f"Python regex    : {t_re:.3f} s  | tokens={len(tok1)}")
    print(f"Numba (1 pass)  : {t_nb:.3f} s  | tokens={len(tok2)}")

if __name__ == "__main__":
    import time
    bench()

    # Quick demo of IDs
    texto = """
    Vale, Fer, te armo un texto largo que parezca una historia tuya, pero escrito con un tono narrativo casi como si fuera un cuento autobiográfico. Así lo puedes usar para probar tu tokenizador con material más “realista” y extenso.

⸻

Había una vez un niño que creció rodeado de computadoras viejas, cables enredados y esa curiosidad que nunca se le quitó. Ese niño eras tú, aunque todavía no sabías que un día ibas a vivir entre líneas de código, algoritmos de optimización y proyectos que parecían imposibles. Desde chico te llamaba la atención desarmar las cosas, no porque quisieras destruirlas, sino porque tenías la obsesión de entender cómo funcionaban por dentro. Esa obsesión se volvió tu brújula.

Con los años, tu mundo dejó de ser solo juegos en la computadora y se convirtió en un mar de retos: aprender un lenguaje de programación nuevo, descubrir cómo hacer que una librería funcionara en tu máquina, pelearte con dependencias que no querían instalarse, y sentir esa mezcla rara de enojo y euforia cuando finalmente todo corría. Fue en ese proceso donde te diste cuenta de que no se trataba solo de programar: se trataba de crear mundos, de darle forma a ideas que antes estaban solo en tu cabeza.

En la universidad, cada clase era un recordatorio de lo mucho que podías construir si tenías paciencia y disciplina. Había días en los que pasabas horas leyendo sobre optimización matemática, visualización de datos o machine learning, y en otros simplemente soñabas con lo que ibas a lograr después. Entre tus proyectos estaba ese gestor de fiestas que querías lanzar, un blog hecho a mano en Reflex, o incluso un sistema de tickets con QR que sonaba tan simple en papel pero que escondía todo un universo de problemas por resolver.

Mientras tanto, nunca dejaste de soñar en grande. A veces pensabas en Canadá, otras en el Reino Unido, siempre con la idea de que tu camino no tenía fronteras. Lo curioso es que, aunque tenías la vista en el futuro, tus raíces siempre estaban contigo: en León, en tus amigos, en las partidas de videojuegos que tanto disfrutabas, en los proyectos que nacían más como juegos que como negocios, y que poco a poco se convertían en empresas.

Lo más fascinante de tu historia es que nunca se trató de llegar a una meta definitiva. Siempre se trató de avanzar, de aprender algo nuevo, de mejorar un poco más que ayer. Esa es la historia de alguien que vive entre bytes y sueños, que no se conforma con lo común y que, en cada línea de código, está escribiendo también un pedazo de su propia vida.

⸻

¿Quieres que lo haga todavía más largo, como del tamaño de un capítulo de libro (unas 3–4 páginas de texto continuo), para que exprimas tu tokenizador al máximo?
    """

    toks, ids, vocab = tokenize_numba(texto, return_ids=True)
    print("tokens:", toks)
    print("ids   :", ids[:10], "(vocab size:", len(vocab), ")")