"""
integration.py
==============
Objetivo 3 — Integração NLP + Análise de Redes

Combina o score do modelo BERT (Objetivo 1) com as métricas de rede
(Objetivo 2) num ensemble ponderado:

    score_final = α × score_nlp + (1 - α) × score_network

O score_network é derivado das métricas estruturais do grafo de propagação,
normalizadas para [0,1] via min-max scaling sobre o conjunto de teste.

Uso:
    python integration.py                        # corre com todos os alphas
    python integration.py --alpha 0.6            # alpha específico
    python integration.py --results_dir results/ # pasta de output customizada
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]

DATA_PROCESSED = ROOT / "data" / "processed" / "fakenewsnet"
RESULTS_METRICS = ROOT / "results" / "metrics"
RESULTS_FIGURES = ROOT / "results" / "figures"

# ficheiros de input
NLP_RESULTS_PATH = RESULTS_METRICS / "predictions_bert.json"   # gerado pelo generate_predictions.py
NETWORK_METRICS_PATH = RESULTS_METRICS / "network_metrics.csv"

# ficheiro de output
INTEGRATION_RESULTS_PATH = RESULTS_METRICS / "results_integration.json"

# ── logger ─────────────────────────────────────────────────────────────────
try:
    from src.FKNWS.utils.get_logger import get_logger
    logger = get_logger("integration")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("integration")

# ── features de rede usadas no score_network ───────────────────────────────
# Selecionadas com base na análise de correlação do Objetivo 2:
#   - degree_centrality_mean  (discriminativa, correlação -0.74 com modularity)
#   - betweenness_mean        (proxy de super-spreaders)
#   - propagation_speed       (fake espalha 43% mais rápido)
#   - modularity              (invertida — fake tem modularity MENOR)
#   - n_communities           (fake tem ligeiramente mais comunidades)
# density e degree_centrality_max removidas (redundantes: corr=1.00 e 0.98)
NETWORK_FEATURES = [
    "degree_centrality_mean",
    "betweenness_mean",
    "propagation_speed",
    "modularity",
    "n_communities",
]

# pesos de cada feature no score_network composto (somam 1.0)
# modularity é invertida (maior modularity → mais real → peso negativo na
# direção "fake"), por isso usamos (1 - modularity_norm) no cálculo
FEATURE_WEIGHTS = {
    "degree_centrality_mean": 0.30,
    "betweenness_mean":       0.25,
    "propagation_speed":      0.25,
    "modularity":             0.10,   # será invertida abaixo
    "n_communities":          0.10,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

def load_nlp_results(path: Path) -> pd.DataFrame:
    """
    Carrega predições do modelo BERT geradas pelo generate_predictions.py.

    Espera a estrutura do predictions_bert.json:
        {
          "predictions": [
            {"article_id": "gossipcop-...", "label": 0|1,
             "pred_nlp": 0|1, "score_nlp": 0.72},
            ...
          ]
        }

    Returns
    -------
    DataFrame com colunas: article_id, label, pred_nlp, score_nlp
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        raise ValueError(f"'predictions' vazio ou ausente em {path}")

    df = pd.DataFrame(preds)

    required = {"article_id", "label", "score_nlp", "pred_nlp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas em falta em predictions_bert.json: {missing}")

    logger.info(f"NLP predictions carregadas: {len(df)} artigos")
    return df[["article_id", "label", "score_nlp", "pred_nlp"]]


def load_network_metrics(path: Path) -> pd.DataFrame:
    """
    Carrega network_metrics.csv (21.693 artigos do Objetivo 2).

    Returns
    -------
    DataFrame com id + NETWORK_FEATURES disponíveis
    """
    df = pd.read_csv(path)

    # network_metrics.csv usa "article_id" como chave
    id_col = "article_id" if "article_id" in df.columns else "id"
    if id_col not in df.columns:
        raise ValueError(f"Coluna de ID não encontrada em {path}")
    if id_col != "article_id":
        df = df.rename(columns={id_col: "article_id"})

    available = [f for f in NETWORK_FEATURES if f in df.columns]
    missing_feats = set(NETWORK_FEATURES) - set(available)
    if missing_feats:
        logger.warning(f"Features de rede em falta (serão ignoradas): {missing_feats}")

    logger.info(f"Network metrics carregadas: {len(df)} artigos, features: {available}")
    return df[["article_id"] + available]


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSTRUÇÃO DO SCORE_NETWORK
# ═══════════════════════════════════════════════════════════════════════════

def build_network_score(df_net: pd.DataFrame) -> pd.Series:
    """
    Constrói score_network ∈ [0,1] a partir das features de rede.

    Processo:
    1. Normaliza cada feature para [0,1] via MinMaxScaler
    2. Inverte modularity (maior modularity → mais real → menor score fake)
    3. Combina com pesos definidos em FEATURE_WEIGHTS

    Parameters
    ----------
    df_net : DataFrame com as features de rede (já filtrado para o subset)

    Returns
    -------
    Series com score_network ∈ [0,1] indexada como df_net
    """
    available_features = [f for f in NETWORK_FEATURES if f in df_net.columns]
    available_weights  = {f: FEATURE_WEIGHTS[f] for f in available_features}

    # re-normalizar pesos para somarem 1.0 (caso features em falta)
    total_w = sum(available_weights.values())
    norm_weights = {f: w / total_w for f, w in available_weights.items()}

    # min-max scaling
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df_net[available_features]),
        columns=available_features,
        index=df_net.index,
    )

    # score composto
    score = pd.Series(0.0, index=df_net.index)
    for feat, w in norm_weights.items():
        if feat == "modularity":
            # modularity alta → rede mais fragmentada → mais real → invertemos
            score += w * (1.0 - scaled[feat])
        else:
            score += w * scaled[feat]

    # garantir [0,1] (segurança numérica)
    score = score.clip(0.0, 1.0)
    logger.info(
        f"score_network — min={score.min():.3f}, max={score.max():.3f}, "
        f"mean={score.mean():.3f}"
    )
    return score


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def compute_ensemble(
    df: pd.DataFrame,
    alpha: float,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Calcula o score final e a predição do ensemble.

    score_final = α × score_nlp + (1 - α) × score_network
    pred_ensemble = 1 (Fake) se score_final >= threshold, else 0 (Real)

    Parameters
    ----------
    df        : DataFrame com colunas score_nlp e score_network
    alpha     : peso do NLP (0 a 1)
    threshold : limiar de decisão (default 0.5)

    Returns
    -------
    df com colunas adicionais: score_final, pred_ensemble
    """
    df = df.copy()
    df["score_final"]    = alpha * df["score_nlp"] + (1 - alpha) * df["score_network"]
    df["pred_ensemble"]  = (df["score_final"] >= threshold).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, y_score=None, label: str = "") -> dict:
    """
    Calcula métricas de classificação standard.

    Returns
    -------
    dict com accuracy, precision, recall, f1, auc (se y_score fornecido)
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_score is not None:
        try:
            metrics["auc"] = round(roc_auc_score(y_true, y_score), 4)
        except ValueError:
            metrics["auc"] = None

    if label:
        logger.info(
            f"[{label}] Acc={metrics['accuracy']} | "
            f"P={metrics['precision']} | R={metrics['recall']} | "
            f"F1={metrics['f1']}"
            + (f" | AUC={metrics['auc']}" if "auc" in metrics else "")
        )
    return metrics


def evaluate_all_approaches(df: pd.DataFrame, alpha: float) -> dict:
    """
    Avalia as 3 abordagens no mesmo subset de teste.

    Returns
    -------
    dict com resultados para baseline_network, nlp, ensemble
    """
    y_true = df["label"].values

    # — só rede (threshold 0.5 no score_network)
    pred_net = (df["score_network"] >= 0.5).astype(int).values
    net_metrics = evaluate(y_true, pred_net, df["score_network"].values, "só rede")

    # — só NLP
    nlp_metrics = evaluate(y_true, df["pred_nlp"].values, df["score_nlp"].values, "só NLP")

    # — ensemble
    ens_metrics = evaluate(
        y_true,
        df["pred_ensemble"].values,
        df["score_final"].values,
        f"ensemble α={alpha}",
    )

    return {
        "network_only":  net_metrics,
        "nlp_only":      nlp_metrics,
        "ensemble":      ens_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. TABELA COMPARATIVA
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    Constrói tabela comparativa de todos os alphas testados.

    Parameters
    ----------
    all_results : {alpha: {"network_only": {...}, "nlp_only": {...}, "ensemble": {...}}}

    Returns
    -------
    DataFrame com uma linha por abordagem/alpha
    """
    rows = []
    # abordagens base (iguais para todos os alphas — usar alpha=0.5)
    base = all_results[list(all_results.keys())[0]]

    rows.append({
        "abordagem": "Só Rede",
        "alpha":     "-",
        **base["network_only"],
    })
    rows.append({
        "abordagem": "Só NLP (BERT)",
        "alpha":     "-",
        **base["nlp_only"],
    })

    for alpha, results in sorted(all_results.items()):
        rows.append({
            "abordagem": f"Ensemble",
            "alpha":     alpha,
            **results["ensemble"],
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run_integration(
    alphas: list[float] | None = None,
    results_dir: Path | None = None,
) -> dict:
    """
    Pipeline completo de integração.

    Parameters
    ----------
    alphas      : lista de alphas a testar (default: [0.5, 0.6, 0.7, 0.8])
    results_dir : pasta onde guardar resultados (default: RESULTS_METRICS)

    Returns
    -------
    dict com todos os resultados e a melhor configuração
    """
    if alphas is None:
        alphas = [0.5, 0.6, 0.7, 0.8]
    if results_dir is None:
        results_dir = RESULTS_METRICS
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("OBJETIVO 3 — INTEGRAÇÃO NLP + REDES")
    logger.info("=" * 60)

    # ── carregar dados ──────────────────────────────────────────────────
    logger.info("A carregar resultados NLP...")
    df_nlp = load_nlp_results(NLP_RESULTS_PATH)

    logger.info("A carregar métricas de rede...")
    df_net = load_network_metrics(NETWORK_METRICS_PATH)

    # ── join por article_id ─────────────────────────────────────────────
    df = df_nlp.merge(df_net, on="article_id", how="inner")
    logger.info(f"Artigos após join NLP ∩ Rede: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            "Join resultou em 0 artigos. Verifica se a coluna 'article_id' é compatível "
            "entre predictions_bert.json e network_metrics.csv."
        )

    # ── score_network ───────────────────────────────────────────────────
    logger.info("A construir score_network...")
    df["score_network"] = build_network_score(df)

    # ── testar alphas ───────────────────────────────────────────────────
    all_results = {}
    for alpha in alphas:
        logger.info(f"\n── α = {alpha} ──")
        df_alpha = compute_ensemble(df, alpha)
        all_results[alpha] = evaluate_all_approaches(df_alpha, alpha)

    # ── melhor alpha (por F1 do ensemble) ───────────────────────────────
    best_alpha = max(
        alphas,
        key=lambda a: all_results[a]["ensemble"]["f1"],
    )
    best_f1 = all_results[best_alpha]["ensemble"]["f1"]
    nlp_f1  = all_results[best_alpha]["nlp_only"]["f1"]
    delta   = round(best_f1 - nlp_f1, 4)

    logger.info("\n" + "=" * 60)
    logger.info(f"MELHOR ALPHA: {best_alpha}  →  F1={best_f1}  (NLP isolado: {nlp_f1}, Δ={delta:+.4f})")
    logger.info("=" * 60)

    # ── tabela comparativa ──────────────────────────────────────────────
    comparison_table = build_comparison_table(all_results)
    table_path = results_dir / "integration_comparison.csv"
    comparison_table.to_csv(table_path, index=False)
    logger.info(f"Tabela comparativa guardada: {table_path}")
    print("\n" + comparison_table.to_string(index=False))

    # ── guardar resultados JSON ─────────────────────────────────────────
    output = {
        "n_test_articles":    len(df),
        "alphas_tested":      alphas,
        "best_alpha":         best_alpha,
        "best_f1_ensemble":   best_f1,
        "nlp_only_f1":        nlp_f1,
        "delta_f1":           delta,
        "all_results":        {str(a): v for a, v in all_results.items()},
    }

    out_path = results_dir / "results_integration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Resultados guardados: {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Integração NLP + Redes (Objetivo 3)")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Alpha específico a testar (ex: 0.6). Se omitido, testa 0.5/0.6/0.7/0.8.",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=None,
        help="Pasta de output (default: results/metrics/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    alphas = [args.alpha] if args.alpha is not None else None
    run_integration(alphas=alphas, results_dir=args.results_dir)