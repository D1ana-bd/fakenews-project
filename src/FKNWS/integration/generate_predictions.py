"""
generate_predictions.py
=======================
Gera predições do modelo BERT (treinado no Objetivo 1) para os artigos
do gossipcop e politifact, usando apenas o título como input.

Estes artigos têm IDs compatíveis com o network_metrics.csv (Objetivo 2),
permitindo o join para o ensemble do Objetivo 3.

Output: results/metrics/predictions_bert.json
    {
      "predictions": [
        {"article_id": "gossipcop-123", "label": 1,
         "pred_nlp": 1, "score_nlp": 0.82},
        ...
      ]
    }

Nota: O BERT foi treinado com título + texto completo (input_text = title [SEP] text).
Aqui usamos apenas o título por limitação do dataset gossipcop/politifact
(que não inclui texto completo). Esta limitação está documentada no relatório.

Uso:
    python generate_predictions.py
    python generate_predictions.py --batch_size 16
    python generate_predictions.py --max_articles 500  # subset para teste rápido
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]

RAW_DATA = ROOT / "data" / "raw" / "fakenewsnet"
RESULTS_METRICS = ROOT / "results" / "metrics"
MODEL_PATH = ROOT / "models" / "bert_fake_news"
NETWORK_METRICS_PATH = RESULTS_METRICS / "network_metrics.csv"
OUTPUT_PATH = RESULTS_METRICS / "predictions_bert.json"

# ── logger ─────────────────────────────────────────────────────────────────
try:
    from src.FKNWS.utils.get_logger import get_logger
    logger = get_logger("generate_predictions")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("generate_predictions")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS
# ═══════════════════════════════════════════════════════════════════════════

def load_gossipcop_politifact() -> pd.DataFrame:
    """
    Carrega os CSVs gossipcop e politifact (fake + real).
    Apenas artigos que estão no network_metrics.csv são incluídos
    (garante que o join posterior funciona).

    Returns
    -------
    DataFrame com colunas: article_id, title, label
        label: 1 = fake, 0 = real
    """
    files = {
        "gossipcop_fake.csv":   1,
        "gossipcop_real.csv":   0,
        "politifact_fake.csv":  1,
        "politifact_real.csv":  0,
    }

    dfs = []
    for filename, label in files.items():
        path = RAW_DATA / filename
        if not path.exists():
            logger.warning(f"Ficheiro não encontrado: {path}")
            continue

        df = pd.read_csv(path, usecols=["id", "title"])
        df = df.rename(columns={"id": "article_id"})
        df["label"] = label
        dfs.append(df)
        logger.info(f"  {filename}: {len(df)} artigos")

    df_all = pd.concat(dfs, ignore_index=True)

    # remover títulos vazios
    df_all = df_all.dropna(subset=["title"])
    df_all = df_all[df_all["title"].str.strip() != ""]

    # filtrar para artigos que estão no network_metrics.csv
    logger.info("A filtrar para artigos com métricas de rede...")
    df_net = pd.read_csv(NETWORK_METRICS_PATH, usecols=["article_id"])
    ids_com_rede = set(df_net["article_id"].values)

    n_antes = len(df_all)
    df_all = df_all[df_all["article_id"].isin(ids_com_rede)]
    logger.info(f"Artigos com métricas de rede: {len(df_all)} / {n_antes}")

    # remover duplicados (mesmo artigo em múltiplos CSVs)
    df_all = df_all.drop_duplicates(subset=["article_id"])

    logger.info(f"Total final: {len(df_all)} artigos únicos")
    logger.info(f"  Fake: {df_all['label'].sum()} | Real: {(df_all['label']==0).sum()}")

    return df_all.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CARREGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_path: Path):
    """
    Carrega o modelo BERT fine-tuned e o tokenizer.

    Returns
    -------
    (model, tokenizer, device)
    """
    logger.info(f"A carregar modelo de {model_path}...")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            "Confirma que os pesos estão em models/bert_fake_news/"
        )

    # device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    logger.info("Modelo carregado com sucesso")
    return model, tokenizer, device


# ═══════════════════════════════════════════════════════════════════════════
# 3. INFERÊNCIA
# ═══════════════════════════════════════════════════════════════════════════

def predict_batch(
    titles: list[str],
    model,
    tokenizer,
    device,
    max_length: int = 128,
) -> tuple[list[int], list[float]]:
    """
    Corre inferência BERT num batch de títulos.

    Parameters
    ----------
    titles     : lista de strings (títulos)
    max_length : comprimento máximo de tokenização (128 suficiente para títulos)

    Returns
    -------
    (preds, scores_fake)
        preds       : lista de 0/1 (Real/Fake)
        scores_fake : lista de probabilidades de ser Fake ∈ [0,1]
    """
    encoding = tokenizer(
        titles,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=-1)  # [batch, 2]

    # índice 1 = Fake (convenção do train_bert.py: 0=Real, 1=Fake)
    scores_fake = probs[:, 1].cpu().tolist()
    preds       = (probs[:, 1] >= 0.5).int().cpu().tolist()

    return preds, scores_fake


def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Corre inferência em todos os artigos em batches.

    Returns
    -------
    df com colunas adicionais: pred_nlp, score_nlp
    """
    all_preds  = []
    all_scores = []

    titles = df["title"].tolist()
    n_batches = (len(titles) + batch_size - 1) // batch_size

    logger.info(f"A correr inferência: {len(titles)} artigos, batch_size={batch_size}")

    for i in tqdm(range(0, len(titles), batch_size), total=n_batches, desc="Inferência BERT"):
        batch_titles = titles[i : i + batch_size]
        preds, scores = predict_batch(batch_titles, model, tokenizer, device)
        all_preds.extend(preds)
        all_scores.extend(scores)

    df = df.copy()
    df["pred_nlp"]  = all_preds
    df["score_nlp"] = all_scores

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. GUARDAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """
    Guarda predições em JSON com estrutura esperada pelo integration.py.
    """
    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            "article_id": row["article_id"],
            "label":      int(row["label"]),
            "pred_nlp":   int(row["pred_nlp"]),
            "score_nlp":  round(float(row["score_nlp"]), 6),
        })

    output = {
        "model":          "bert_finetuned_cascade",
        "input":          "title_only",
        "n_articles":     len(predictions),
        "predictions":    predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Predições guardadas: {output_path} ({len(predictions)} artigos)")

    # estatísticas rápidas
    df_check = pd.DataFrame(predictions)
    from sklearn.metrics import f1_score, accuracy_score
    acc = accuracy_score(df_check["label"], df_check["pred_nlp"])
    f1  = f1_score(df_check["label"], df_check["pred_nlp"], zero_division=0)
    logger.info(f"Accuracy={acc:.4f} | F1={f1:.4f} (títulos apenas — esperado < 0.52)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run(batch_size: int = 16, max_articles: int | None = None) -> None:
    logger.info("=" * 60)
    logger.info("GENERATE PREDICTIONS — BERT em gossipcop/politifact")
    logger.info("=" * 60)

    # carregar dados
    df = load_gossipcop_politifact()

    if max_articles is not None:
        df = df.sample(n=min(max_articles, len(df)), random_state=42)
        logger.info(f"Subset de teste: {len(df)} artigos (--max_articles={max_articles})")

    # carregar modelo
    model, tokenizer, device = load_model(MODEL_PATH)

    # inferência
    df = run_inference(df, model, tokenizer, device, batch_size=batch_size)

    # guardar
    save_predictions(df, OUTPUT_PATH)

    logger.info(" Concluído! Próximo passo: correr integration.py")


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera predições BERT para artigos gossipcop/politifact"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Tamanho do batch para inferência (default: 16)",
    )
    parser.add_argument(
        "--max_articles", type=int, default=None,
        help="Limitar a N artigos (útil para teste rápido)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(batch_size=args.batch_size, max_articles=args.max_articles)