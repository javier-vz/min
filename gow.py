# -------------------------------------------------------------------------
# UDHR → CLEAN CORPORA → GRAPH-OF-WORDS (GoW) PER LANGUAGE
# -------------------------------------------------------------------------
# Reads languages_and_dialects_geo.csv, loads UDHR texts,
# cleans and tokenizes them, builds weighted co-occurrence graphs,
# and stores them in pickled dictionaries for later use.
# -------------------------------------------------------------------------

import os
import re
import pickle
import pandas as pd
import networkx as nx
from collections import defaultdict

# -------------------------------------------------------------------------
# LOAD MACROAREA AND ISO CODES
# -------------------------------------------------------------------------
MACROAREA_CSV = "languages_and_dialects_geo.csv"
UDHR_DIR = "udhr"
OUT_DIR = "pickles"

os.makedirs(OUT_DIR, exist_ok=True)

macroarea_df = pd.read_csv(MACROAREA_CSV, sep=",")
macroarea_df.dropna(inplace=True)
macroarea_df = macroarea_df[["isocodes", "macroarea"]]
macroarea = dict(zip(macroarea_df["isocodes"], macroarea_df["macroarea"]))

# -------------------------------------------------------------------------
# READ UDHR FILES PER LANGUAGE
# -------------------------------------------------------------------------
languages: dict[str, list[str]] = {}

for language in macroarea.keys():
    path = os.path.join(UDHR_DIR, f"udhr_{language}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().split("\n")
        lines = [line.strip() for line in text if len(line.strip()) > 0]
        languages[language] = lines
    except FileNotFoundError:
        continue

# -------------------------------------------------------------------------
# TOKENIZE AND CLEAN
# -------------------------------------------------------------------------
def tokenize(s: str) -> list[str]:
    return s.split(" ")

def clean(L: str) -> list[list[str]]:
    sentences = languages[L]
    table = str.maketrans({k: None for k in '``!"#$%&\\¿()*+,-./:;<=>?@[\\]_{|}'})
    cleaned = []
    for sentence in sentences:
        s0 = sentence.translate(table)
        toks = [w.lower().translate(table) for w in tokenize(s0)]
        toks = [w for w in toks if w and w not in ("''", "̃") and not w.isdigit()]
        if toks:
            cleaned.append(toks)
    if L == "zro":
        return cleaned[6:]
    elif L == "tca":
        return cleaned[7:]
    elif L == "gyr":
        return cleaned[9:]
    else:
        return cleaned[5:]

clean_languages: dict[str, list[list[str]]] = {}
for L in languages:
    clean_languages[L] = clean(L)

# -------------------------------------------------------------------------
# GRAPH-OF-WORDS (GoW)
# -------------------------------------------------------------------------
def GoW(tokens_or_sentences, window_size: int = 2) -> nx.Graph:
    if tokens_or_sentences and isinstance(tokens_or_sentences[0], list):
        tokens = [t for sent in tokens_or_sentences for t in sent]
    else:
        tokens = list(tokens_or_sentences)
    cooc = defaultdict(int)
    n = len(tokens)
    for i, w1 in enumerate(tokens):
        for j in range(i + 1, min(i + window_size + 1, n)):
            w2 = tokens[j]
            if w1 == w2:
                continue
            a, b = (w1, w2) if w1 <= w2 else (w2, w1)
            cooc[(a, b)] += 1
    G = nx.Graph()
    for (w1, w2), w in cooc.items():
        G.add_edge(w1, w2, weight=w)
    return G

graphs: dict[str, nx.Graph] = {}
for language in languages.keys():
    graphs[language] = GoW(clean_languages[language])

# -------------------------------------------------------------------------
# SAVE DICTIONARIES AS PICKLES
# -------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "languages.pkl"), "wb") as f:
    pickle.dump(languages, f)
with open(os.path.join(OUT_DIR, "clean_languages.pkl"), "wb") as f:
    pickle.dump(clean_languages, f)
with open(os.path.join(OUT_DIR, "graphs.pkl"), "wb") as f:
    pickle.dump(graphs, f)

print(f"✅ Saved pickle files to '{OUT_DIR}/':")
print("   - languages.pkl")
print("   - clean_languages.pkl")
print("   - graphs.pkl")
