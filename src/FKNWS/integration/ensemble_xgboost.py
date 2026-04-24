"""
ensemble_xgboost.py
===================
Ensemble XGBoost combinando features NLP + Análise de Redes.

Substitui o ensemble linear (α fixo) por um classificador que aprende
automaticamente os pesos ótimos das features.

Testa 3 abordagens NLP:
    Abordagem 1: BERT atual (treinado título+texto, aplicado a títulos)
    Abordagem 2: Zero-shot (sem treino)
    Abordagem 3: BERT re-treinado só com títulos (TODO: fase seguinte)

Features do ensemble:
    score_nlp               ← score do módulo NLP
    degree_centrality_mean  ← centralidade média
    betweenness_mean        ← betweenness médio
    propagation_speed       ← velocidade de propagação
    modularity              ← modularidade (invertida = mais fake)
    n_communities           ← número de comunidades

Uso:
    python src/FKNWS/integration/ensemble_xgboost.py
    python src/FKNWS/integration/ensemble_xgboost.py --approach all
    python src/FKNWS/integration/ensemble_xgboost.py --approach zeroshot
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# ── paths ───────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[3]
METRICS_DIR      = ROOT / "results" / "metrics"

BERT_PREDS_PATH  = METRICS_DIR / "predictions_bert.json"
ZSHOT_PREDS_PATH = METRICS_DIR / "results_zeroshot.json"
NETWORK_PATH     = METRICS_DIR / "network_metrics.csv"
OUTPUT_PATH      = METRICS_DIR / "results_xgboost.json"

# features de rede (mesmas do integration.py)
NETWORK_FEATURES = [
    "degree_centrality_mean",
    "betweenness_mean",
    "propagation_speed",
    "modularity",
    "n_communities",
]

# ── logger ───────────────────────────────────────────────────────────────────
try:
    from src.FKNWS.utils.get_logger import get_logger
    logger = get_logger("ensemble_xgboost")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("ensemble_xgboost")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

def load_bert_predictions(path: Path) -> pd.DataFrame:
    """
    Carrega predictions_bert.json (Abordagem 1).

    Returns
    -------
    DataFrame com colunas: article_id, label, score_nlp_bert
    """
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["predictions"])
    df = df.rename(columns={"score_nlp": "score_nlp_bert"})
    logger.info(f"BERT predictions carregadas: {len(df)} artigos")
    return df[["article_id", "label", "score_nlp_bert"]]


def load_zeroshot_predictions(path: Path) -> pd.DataFrame:
    """
    Carrega results_zeroshot.json (Abordagem 2).

    Nota: o zero-shot foi avaliado no test.csv (43 artigos BuzzFeed/PolitiFact)
    que NÃO tem métricas de rede. Por isso criamos scores sintéticos para
    demonstração da arquitetura XGBoost.

    Para ensemble real, o zero-shot seria aplicado aos títulos do gossipcop
    (que têm métricas de rede) — implementado na função run_zeroshot_on_gossipcop.

    Returns
    -------
    DataFrame com colunas: article_id, label, score_nlp_zeroshot
    """
    with open(path) as f:
        data = json.load(f)

    preds = data["predictions"]
    df = pd.DataFrame([{
        "title":               p["title"],
        "label":               p["label"],
        "score_nlp_zeroshot":  p["score_fake"],
    } for p in preds])

    logger.info(f"Zero-shot predictions carregadas: {len(df)} artigos")
    return df


def load_network_metrics(path: Path) -> pd.DataFrame:
    """
    Carrega network_metrics.csv.

    Returns
    -------
    DataFrame com article_id + NETWORK_FEATURES
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["label"], errors="ignore")

    available = [f for f in NETWORK_FEATURES if f in df.columns]
    missing   = set(NETWORK_FEATURES) - set(available)
    if missing:
        logger.warning(f"Features em falta: {missing}")

    logger.info(f"Network metrics carregadas: {len(df)} artigos")
    return df[["article_id"] + available]


# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPARAÇÃO DAS FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def prepare_features(
    df_nlp: pd.DataFrame,
    df_net: pd.DataFrame,
    nlp_col: str,
    join_col: str = "article_id",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Junta NLP scores com features de rede e normaliza.

    Parameters
    ----------
    df_nlp   : DataFrame com join_col, label, nlp_col
    df_net   : DataFrame com article_id + NETWORK_FEATURES
    nlp_col  : nome da coluna do score NLP
    join_col : coluna de join

    Returns
    -------
    (X, y) — features normalizadas e labels
    """
    df = df_nlp.merge(df_net, on=join_col, how="inner")
    logger.info(f"Artigos após join: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            f"Join resultou em 0 artigos. "
            f"Verifica se '{join_col}' é compatível entre os dois DataFrames."
        )

    available_net = [f for f in NETWORK_FEATURES if f in df.columns]
    feature_cols  = [nlp_col] + available_net

    # normalizar features de rede (NLP já está em [0,1])
    scaler = MinMaxScaler()
    df[available_net] = scaler.fit_transform(df[available_net])

    X = df[feature_cols]
    y = df["label"]

    logger.info(f"Features: {feature_cols}")
    logger.info(f"  Fake: {y.sum()} | Real: {(y==0).sum()}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 3. TREINO E AVALIAÇÃO XGBOOST
# ═══════════════════════════════════════════════════════════════════════════

def train_evaluate_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    approach_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Treina e avalia XGBoost com cross-validation estratificada.

    Usa cross-validation porque o dataset é pequeno (21k artigos mas
    com desbalanceamento 74%/26%) — CV dá estimativa mais robusta.

    Parameters
    ----------
    X             : features
    y             : labels
    approach_name : nome da abordagem (para logging)
    n_splits      : número de folds CV
    random_state  : seed

    Returns
    -------
    dict com métricas + feature_importance
    """
    # calcular scale_pos_weight para lidar com desbalanceamento
    n_fake = y.sum()
    n_real = (y == 0).sum()
    scale_pos_weight = n_real / n_fake if n_fake > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # cross-validation AUC
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1")

    logger.info(
        f"[{approach_name}] CV AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f} | "
        f"F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}"
    )

    # treinar modelo final em todos os dados para feature importance
    model.fit(X, y)
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # métricas no conjunto completo
    metrics = {
        "cv_auc_mean":  round(float(auc_scores.mean()), 4),
        "cv_auc_std":   round(float(auc_scores.std()),  4),
        "cv_f1_mean":   round(float(f1_scores.mean()),  4),
        "cv_f1_std":    round(float(f1_scores.std()),   4),
        "train_accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        "train_precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "train_recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "train_f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "train_auc":       round(float(roc_auc_score(y, y_proba)), 4),
    }

    # feature importance
    importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    importance = {k: round(float(v), 4)
                  for k, v in sorted(importance.items(),
                                     key=lambda x: x[1], reverse=True)}

    logger.info(f"  Feature importance: {importance}")

    return {
        "metrics":            metrics,
        "feature_importance": importance,
        "scale_pos_weight":   round(float(scale_pos_weight), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run(approach: str = "all") -> dict:
    """
    Pipeline completo do ensemble XGBoost.

    Parameters
    ----------
    approach : "bert" | "zeroshot" | "all"
    """
    logger.info("=" * 60)
    logger.info("ENSEMBLE XGBOOST — NLP + FEATURES DE REDE")
    logger.info("=" * 60)

    # carregar network metrics
    df_net = load_network_metrics(NETWORK_PATH)

    results = {}

    # ── Abordagem 1: BERT atual ──────────────────────────────────────────
    if approach in ("bert", "all"):
        logger.info("\n── Abordagem 1: BERT atual + XGBoost ──")
        df_bert = load_bert_predictions(BERT_PREDS_PATH)
        X_bert, y_bert = prepare_features(
            df_bert, df_net,
            nlp_col="score_nlp_bert",
            join_col="article_id",
        )
        results["abordagem_1_bert"] = train_evaluate_xgboost(
            X_bert, y_bert, "BERT + XGBoost"
        )

    # ── Abordagem 2: Zero-shot ───────────────────────────────────────────
    if approach in ("zeroshot", "all"):
        logger.info("\n── Abordagem 2: Zero-shot + XGBoost ──")
        logger.info(
            "NOTA: Zero-shot avaliado no test.csv (BuzzFeed/PolitiFact).\n"
            "      Para ensemble completo, aplicar zero-shot aos títulos\n"
            "      do gossipcop/politifact (que têm métricas de rede).\n"
            "      A demonstrar aqui: arquitetura XGBoost com score NLP."
        )

        # aplicar zero-shot diretamente aos títulos do gossipcop
        # (que têm métricas de rede)
        df_zshot = run_zeroshot_on_gossipcop(df_net)
        if df_zshot is not None and len(df_zshot) > 0:
            X_zshot, y_zshot = prepare_features(
                df_zshot, df_net,
                nlp_col="score_nlp_zeroshot",
                join_col="article_id",
            )
            results["abordagem_2_zeroshot"] = train_evaluate_xgboost(
                X_zshot, y_zshot, "Zero-shot + XGBoost"
            )
        else:
            logger.warning("Zero-shot no gossipcop não disponível — a saltar.")

    # ── comparação final ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("COMPARAÇÃO FINAL — XGBoost vs Ensemble Linear")
    logger.info("=" * 60)
    logger.info(f"  Ensemble linear (α=0.5) AUC : 0.6718  (referência)")

    for name, res in results.items():
        auc  = res["metrics"]["cv_auc_mean"]
        f1   = res["metrics"]["cv_f1_mean"]
        delta = round(auc - 0.6718, 4)
        logger.info(f"  {name:<30} AUC={auc} | F1={f1} | Δ AUC={delta:+.4f}")
    logger.info("=" * 60)

    # guardar
    output = {
        "ensemble_type":        "XGBoost",
        "network_features":     NETWORK_FEATURES,
        "reference_linear_auc": 0.6718,
        "results":              results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Resultados guardados: {OUTPUT_PATH}")

    return output


def run_zeroshot_on_gossipcop(df_net: pd.DataFrame) -> pd.DataFrame | None:
    """
    Aplica zero-shot classification aos títulos do gossipcop/politifact
    para obter scores NLP compatíveis com as métricas de rede.

    Esta função resolve o problema de incompatibilidade de datasets:
    o zero-shot não precisa de treino, por isso pode ser aplicado
    diretamente aos títulos do gossipcop.

    Returns
    -------
    DataFrame com article_id, label, score_nlp_zeroshot
    ou None se falhar
    """
    import torch
    from transformers import pipeline as hf_pipeline

    RAW_DATA = ROOT / "data" / "raw" / "fakenewsnet"

    try:
        # carregar títulos do gossipcop/politifact
        files = {
            "gossipcop_fake.csv":  1,
            "gossipcop_real.csv":  0,
            "politifact_fake.csv": 1,
            "politifact_real.csv": 0,
        }

        dfs = []
        for fname, label in files.items():
            path = RAW_DATA / fname
            if path.exists():
                df = pd.read_csv(path, usecols=["id", "title"])
                df = df.rename(columns={"id": "article_id"})
                df["label"] = label
                dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.dropna(subset=["title"])
        df_all = df_all[df_all["title"].str.strip() != ""]
        df_all = df_all.drop_duplicates(subset=["article_id"])

        # filtrar para artigos com métricas de rede
        ids_com_rede = set(df_net["article_id"].values)
        df_all = df_all[df_all["article_id"].isin(ids_com_rede)]
        logger.info(f"Artigos para zero-shot (gossipcop): {len(df_all)}")

        if len(df_all) == 0:
            return None

        # carregar pipeline zero-shot
        device = 0 if torch.cuda.is_available() else -1
        classifier = hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=device,
        )

        from FKNWS.models.zero_shot import (
            CANDIDATE_LABELS,
            classify_titles,
            convert_to_score,
        )

        # classificar em batches
        titles  = df_all["title"].tolist()
        results = classify_titles(titles, classifier, batch_size=16)
        scores  = [convert_to_score(r) for r in results]

        df_all["score_nlp_zeroshot"] = scores
        logger.info(
            f"Zero-shot gossipcop concluído: "
            f"mean_fake={df_all[df_all['label']==1]['score_nlp_zeroshot'].mean():.3f} | "
            f"mean_real={df_all[df_all['label']==0]['score_nlp_zeroshot'].mean():.3f}"
        )

        return df_all[["article_id", "label", "score_nlp_zeroshot"]]

    except Exception as e:
        logger.error(f"Erro no zero-shot gossipcop: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble XGBoost — NLP + features de rede"
    )
    parser.add_argument(
        "--approach",
        choices=["bert", "zeroshot", "all"],
        default="all",
        help="Abordagem NLP a usar (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(approach=args.approach)