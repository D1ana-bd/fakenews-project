"""
zero_shot.py
============
Abordagem 2 — Zero-Shot Classification para deteção de fake news.

Usa um modelo NLI (Natural Language Inference) pré-treinado para classificar
títulos de notícias como fake ou real SEM necessidade de fine-tuning.

Modelo: cross-encoder/nli-deberta-v3-small
Input:  title_clean do test.csv
Output: results/metrics/results_zeroshot.json

Uso:
    python src/FKNWS/models/zero_shot.py
    python src/FKNWS/models/zero_shot.py --batch_size 8
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# ── paths ───────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[3]
TEST_CSV_PATH   = ROOT / "data" / "processed" / "fakenewsnet" / "test.csv"
RESULTS_PATH    = ROOT / "results" / "metrics" / "results_zeroshot.json"

# ── modelo ───────────────────────────────────────────────────────────────────
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# labels — descrição semântica rica melhora o zero-shot
CANDIDATE_LABELS = [
    "fake news or misinformation with misleading claims",
    "factual and verified news reporting",
]
FAKE_LABEL = CANDIDATE_LABELS[0]  # índice do label fake

# ── logger ───────────────────────────────────────────────────────────────────
try:
    from FKNWS.utils.get_logger import get_logger
    logger = get_logger("zero_shot")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("zero_shot")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

def load_test_data(path: Path) -> pd.DataFrame:
    """
    Carrega o test.csv e filtra títulos vazios.

    Returns
    -------
    DataFrame com colunas: id, title_clean, label
    """
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")

    df = pd.read_csv(path)

    required = {"id", "title_clean", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas em falta: {missing}")

    # remover títulos vazios
    n_before = len(df)
    df = df.dropna(subset=["title_clean"])
    df = df[df["title_clean"].str.strip() != ""]
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning(f"{n_removed} títulos vazios removidos")

    logger.info(f"Test data carregada: {len(df)} artigos")
    logger.info(f"  Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICAÇÃO ZERO-SHOT
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_score(result: dict) -> float:
    """
    Converte o output do pipeline zero-shot para score_fake ∈ [0,1].

    O pipeline devolve labels ordenados por score decrescente.
    Precisamos de encontrar o índice do label fake independentemente
    da ordem.

    Parameters
    ----------
    result : dict com 'labels' e 'scores'

    Returns
    -------
    score_fake ∈ [0,1]
    """
    labels = result["labels"]
    scores = result["scores"]

    # encontrar índice do label fake
    fake_idx = next(
        (i for i, l in enumerate(labels) if "fake" in l.lower()),
        0  # default: primeiro label
    )
    return float(scores[fake_idx])


def classify_titles(
    titles: list[str],
    pipeline,
    batch_size: int = 8,
) -> list[dict]:
    """
    Classifica uma lista de títulos usando o pipeline zero-shot.

    Parameters
    ----------
    titles     : lista de strings
    pipeline   : pipeline zero-shot da HuggingFace
    batch_size : tamanho do batch

    Returns
    -------
    lista de dicts com 'labels' e 'scores'
    """
    if not titles:
        return []

    all_results = []

    for i in tqdm(range(0, len(titles), batch_size),
                  desc="Zero-shot classification"):
        batch = titles[i: i + batch_size]
        batch_results = pipeline(
            batch,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False,
        )
        # pipeline devolve lista se input for lista
        if isinstance(batch_results, dict):
            batch_results = [batch_results]
        all_results.extend(batch_results)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 3. AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_zeroshot(
    y_true: list[int],
    results: list[dict],
    threshold: float = 0.5,
) -> dict:
    """
    Avalia as predições zero-shot.

    Parameters
    ----------
    y_true    : labels reais (0=Real, 1=Fake)
    results   : lista de dicts do pipeline
    threshold : limiar de decisão

    Returns
    -------
    dict com accuracy, precision, recall, f1, auc
    """
    scores_fake = [convert_to_score(r) for r in results]
    y_pred      = [1 if s >= threshold else 0 for s in scores_fake]

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    try:
        metrics["auc"] = round(roc_auc_score(y_true, scores_fake), 4)
    except ValueError:
        metrics["auc"] = None

    logger.info(
        f"[Zero-shot] Acc={metrics['accuracy']} | "
        f"P={metrics['precision']} | R={metrics['recall']} | "
        f"F1={metrics['f1']} | AUC={metrics['auc']}"
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 4. GUARDAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

def save_results(
    metrics: dict,
    y_true: list[int],
    results: list[dict],
    titles: list[str],
    path: Path,
) -> None:
    """
    Guarda resultados em JSON.

    Estrutura:
        {
          "model": "cross-encoder/nli-deberta-v3-small",
          "approach": "zero-shot",
          "candidate_labels": [...],
          "metrics": {...},
          "n_articles": N,
          "predictions": [
            {"title": "...", "label": 1, "pred": 1, "score_fake": 0.82},
            ...
          ]
        }
    """
    scores_fake = [convert_to_score(r) for r in results]
    y_pred      = [1 if s >= 0.5 else 0 for s in scores_fake]

    predictions = [
        {
            "title":      title,
            "label":      int(label),
            "pred":       int(pred),
            "score_fake": round(float(score), 6),
        }
        for title, label, pred, score in zip(titles, y_true, y_pred, scores_fake)
    ]

    output = {
        "model":            MODEL_NAME,
        "approach":         "zero-shot",
        "candidate_labels": CANDIDATE_LABELS,
        "metrics":          metrics,
        "n_articles":       len(predictions),
        "predictions":      predictions,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Resultados guardados: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run(batch_size: int = 8) -> dict:
    """Pipeline completo zero-shot."""
    import torch
    from transformers import pipeline

    logger.info("=" * 60)
    logger.info("ABORDAGEM 2 — ZERO-SHOT CLASSIFICATION")
    logger.info("=" * 60)

    # device
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Device: {'cuda' if device == 0 else 'cpu'}")

    # carregar dados
    df = load_test_data(TEST_CSV_PATH)

    # carregar pipeline
    logger.info(f"A carregar modelo: {MODEL_NAME}")
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )

    # classificar
    titles = df["title_clean"].tolist()
    results = classify_titles(titles, classifier, batch_size=batch_size)

    # avaliar
    y_true  = df["label"].tolist()
    metrics = evaluate_zeroshot(y_true, results)

    # comparação com BERT atual
    logger.info("\n" + "=" * 60)
    logger.info("COMPARAÇÃO — Zero-shot vs BERT fine-tuned")
    logger.info("=" * 60)
    logger.info(f"  Zero-shot F1  : {metrics['f1']}")
    logger.info(f"  BERT cascata  : 0.5217  (referência Objetivo 1)")
    logger.info(f"  Diferença F1  : {round(metrics['f1'] - 0.5217, 4):+.4f}")
    logger.info("=" * 60)

    # guardar
    save_results(metrics, y_true, results, titles, RESULTS_PATH)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot classification para fake news (Abordagem 2)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Tamanho do batch (default: 8)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(batch_size=args.batch_size)