"""
train_bert_titles.py
====================
Abordagem 3 — BERT/DistilBERT fine-tuned exclusivamente em títulos,
com data augmentation para aumentar os dados de treino.

Problema que resolve:
    O BERT original foi treinado com título+texto mas o ensemble
    usa apenas títulos (gossipcop não tem texto completo).
    Re-treinar só com títulos elimina o mismatch treino/inferência.

Data Augmentation:
    Para cada artigo do train.csv, cria múltiplos exemplos:
    - Tipo A: só título (simula o cenário de inferência real)
    - Tipo B: título + primeiros 50 words do texto
    - Tipo C: título + primeiros 100 words do texto

Modelos testados:
    - bert-base-uncased     (110M params, mais poderoso)
    - distilbert-base-uncased (66M params, 40% mais rápido)

Uso:
    python src/FKNWS/models/train_bert_titles.py
    python src/FKNWS/models/train_bert_titles.py --model distilbert
    python src/FKNWS/models/train_bert_titles.py --model bert
    python src/FKNWS/models/train_bert_titles.py --model both
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

# ── paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[3]
DATA_DIR      = ROOT / "data" / "processed" / "fakenewsnet"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results" / "metrics"

TRAIN_CSV     = DATA_DIR / "train.csv"
VAL_CSV       = DATA_DIR / "val.csv"
TEST_CSV      = DATA_DIR / "test.csv"

# model configs
MODEL_CONFIGS = {
    "bert": {
        "name":       "bert-base-uncased",
        "output_dir": MODELS_DIR / "bert_titles",
        "results":    RESULTS_DIR / "results_bert_titles.json",
        "tokenizer_class": BertTokenizer,
        "model_class":     BertForSequenceClassification,
    },
    "distilbert": {
        "name":       "distilbert-base-uncased",
        "output_dir": MODELS_DIR / "distilbert_titles",
        "results":    RESULTS_DIR / "results_distilbert_titles.json",
        "tokenizer_class": DistilBertTokenizer,
        "model_class":     DistilBertForSequenceClassification,
    },
}

# ── logger ───────────────────────────────────────────────────────────────────
try:
    from FKNWS.utils.get_logger import get_logger
    logger = get_logger("train_bert_titles")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("train_bert_titles")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def augment_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria múltiplos exemplos por artigo usando data augmentation.

    Para cada artigo gera:
        Tipo A: só título
        Tipo B: título + primeiros 50 words do texto
        Tipo C: título + primeiros 100 words do texto

    Parameters
    ----------
    df : DataFrame com title_clean, text_clean, label

    Returns
    -------
    DataFrame aumentado com colunas: input_text, label, aug_type
    """
    rows = []

    for _, row in df.iterrows():
        title = str(row["title_clean"]).strip()
        text  = str(row["text_clean"]).strip() if pd.notna(row["text_clean"]) else ""
        label = int(row["label"])

        words = text.split()

        # Tipo A — só título (cenário real de inferência)
        rows.append({
            "input_text": title,
            "label":      label,
            "aug_type":   "title_only",
        })

        # Tipo B — título + primeiros 50 words
        if len(words) >= 10:
            snippet_50 = " ".join(words[:50])
            rows.append({
                "input_text": f"{title} [SEP] {snippet_50}",
                "label":      label,
                "aug_type":   "title_50w",
            })

        # Tipo C — título + primeiros 100 words
        if len(words) >= 50:
            snippet_100 = " ".join(words[:100])
            rows.append({
                "input_text": f"{title} [SEP] {snippet_100}",
                "label":      label,
                "aug_type":   "title_100w",
            })

    result = pd.DataFrame(rows)
    logger.info(
        f"Data augmentation: {len(df)} artigos → {len(result)} exemplos "
        f"({len(result)/len(df):.1f}x)"
    )
    logger.info(
        f"  title_only: {(result['aug_type']=='title_only').sum()} | "
        f"title_50w: {(result['aug_type']=='title_50w').sum()} | "
        f"title_100w: {(result['aug_type']=='title_100w').sum()}"
    )
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════════════

class TitlesDataset(Dataset):
    """Dataset PyTorch para títulos (com ou sem augmentation)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. TREINO
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    """Treina uma epoch e devolve a loss média."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model, loader, device) -> tuple[float, float]:
    """Avalia o modelo e devolve (loss, f1)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            probs = F.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


# ═══════════════════════════════════════════════════════════════════════════
# 4. AVALIAÇÃO FINAL NO TEST SET
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_on_test(model, tokenizer, df_test: pd.DataFrame, device) -> dict:
    """
    Avalia o modelo no test.csv usando só títulos (simula inferência real).

    Returns
    -------
    dict com accuracy, precision, recall, f1, auc
    """
    model.eval()
    titles = df_test["title_clean"].tolist()
    labels = df_test["label"].tolist()

    all_scores = []
    all_preds  = []

    batch_size = 16
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i+batch_size]
        enc = tokenizer(
            batch_titles,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            probs   = F.softmax(outputs.logits, dim=-1)

        scores = probs[:, 1].cpu().tolist()
        preds  = (probs[:, 1] >= 0.5).int().cpu().tolist()
        all_scores.extend(scores)
        all_preds.extend(preds)

    metrics = {
        "accuracy":  round(accuracy_score(labels, all_preds), 4),
        "precision": round(precision_score(labels, all_preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, all_preds, zero_division=0), 4),
        "f1":        round(f1_score(labels, all_preds, zero_division=0), 4),
    }
    try:
        metrics["auc"] = round(roc_auc_score(labels, all_scores), 4)
    except ValueError:
        metrics["auc"] = None

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE DE TREINO
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model_key: str,
    lr: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 5,
    patience: int = 2,
    random_state: int = 42,
) -> dict:
    """
    Pipeline completo de treino para um modelo.

    Parameters
    ----------
    model_key    : "bert" ou "distilbert"
    lr           : learning rate
    batch_size   : tamanho do batch
    epochs       : número máximo de epochs
    patience     : early stopping patience
    random_state : seed

    Returns
    -------
    dict com métricas de treino e teste
    """
    config = MODEL_CONFIGS[model_key]
    torch.manual_seed(random_state)

    logger.info("=" * 60)
    logger.info(f"ABORDAGEM 3 — {config['name'].upper()} TÍTULOS + DATA AUG")
    logger.info("=" * 60)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # carregar dados
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # data augmentation no treino
    df_train_aug = augment_titles(df_train)

    # val e test: só título (sem augmentation — simula inferência real)
    df_val_titles = pd.DataFrame({
        "input_text": df_val["title_clean"].astype(str),
        "label":      df_val["label"],
    })

    # carregar tokenizer e modelo
    logger.info(f"A carregar modelo: {config['name']}")
    tokenizer = config["tokenizer_class"].from_pretrained(config["name"])
    model     = config["model_class"].from_pretrained(
        config["name"], num_labels=2
    )
    model.to(device)

    # datasets e dataloaders
    train_dataset = TitlesDataset(
        df_train_aug["input_text"].tolist(),
        df_train_aug["label"].tolist(),
        tokenizer,
    )
    val_dataset = TitlesDataset(
        df_val_titles["input_text"].tolist(),
        df_val_titles["label"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # optimizer e scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # treino com early stopping
    best_val_f1    = 0.0
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    logger.info(
        f"Treino: {len(train_dataset)} exemplos (aug) | "
        f"Val: {len(val_dataset)} exemplos | "
        f"Batch: {batch_size} | LR: {lr}"
    )

    for epoch in range(1, epochs + 1):
        train_loss         = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_f1   = evaluate_epoch(model, val_loader, device)

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_f1":     round(val_f1, 4),
        })

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        # early stopping por val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1   = val_f1
            patience_count = 0
            # guardar melhor modelo
            config["output_dir"].mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config["output_dir"])
            tokenizer.save_pretrained(config["output_dir"])
            logger.info(f"  ✅ Melhor modelo guardado (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            logger.info(f"  ❌ Sem melhoria ({patience_count}/{patience})")
            if patience_count >= patience:
                logger.info("Early stopping ativado!")
                break

    # avaliar no test set com o melhor modelo
    logger.info("A avaliar no test set...")
    best_model = config["model_class"].from_pretrained(config["output_dir"])
    best_model.to(device)
    test_metrics = evaluate_on_test(best_model, tokenizer, df_test, device)

    logger.info(
        f"TEST — Acc={test_metrics['accuracy']} | "
        f"P={test_metrics['precision']} | "
        f"R={test_metrics['recall']} | "
        f"F1={test_metrics['f1']} | "
        f"AUC={test_metrics['auc']}"
    )

    # comparação com BERT original
    logger.info("\n" + "=" * 60)
    logger.info(f"COMPARAÇÃO — {model_key.upper()} títulos vs BERT original")
    logger.info("=" * 60)
    logger.info(f"  {model_key} títulos+aug F1 : {test_metrics['f1']}")
    logger.info(f"  BERT cascata F1        : 0.5217  (referência)")
    logger.info(f"  Diferença F1           : {round(test_metrics['f1'] - 0.5217, 4):+.4f}")
    logger.info("=" * 60)

    # guardar resultados
    output = {
        "model":          config["name"],
        "approach":       f"{model_key}_titles_augmented",
        "config": {
            "lr":         lr,
            "batch_size": batch_size,
            "epochs":     epochs,
            "patience":   patience,
            "aug_types":  ["title_only", "title_50w", "title_100w"],
        },
        "training_history": history,
        "best_val_loss":    round(best_val_loss, 4),
        "best_val_f1":      round(best_val_f1, 4),
        "test_metrics":     test_metrics,
        "comparison": {
            "bert_cascade_f1": 0.5217,
            "delta_f1":        round(test_metrics["f1"] - 0.5217, 4),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config["results"], "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Resultados guardados: {config['results']}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Treino BERT/DistilBERT só com títulos + data augmentation"
    )
    parser.add_argument(
        "--model",
        choices=["bert", "distilbert", "both"],
        default="both",
        help="Modelo a treinar (default: both)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Número máximo de epochs (default: 5)",
    )
    parser.add_argument(
        "--patience", type=int, default=2,
        help="Early stopping patience (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "both":
        models_to_train = ["distilbert", "bert"]
    else:
        models_to_train = [args.model]

    for model_key in models_to_train:
        train_model(
            model_key=model_key,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
        )