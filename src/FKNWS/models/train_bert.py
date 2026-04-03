"""
train_bert.py
─────────────
Script de fine-tuning BERT em cascata para deteção de fake news.

Estratégia:
    Fase 1 — Pré-fine-tune no LIAR (12k exemplos, domínio geral)
    Fase 2 — Fine-tune final no FakeNewsNet (422 exemplos, domínio artigos)

Uso:
    python src/FKNWS/models/train_bert.py --phase 1   # treinar no LIAR
    python src/FKNWS/models/train_bert.py --phase 2   # fine-tune no FakeNewsNet
    python src/FKNWS/models/train_bert.py --phase all # ambas em sequência
    python src/FKNWS/models/train_bert.py --test_run  # subset 10% para testar pipeline

"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.FKNWS.utils.get_logger import get_logger

logger = get_logger("train_bert")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[3]
LIAR_DIR       = PROJECT_ROOT / "data" / "raw" / "LIAR"
FKN_DIR        = PROJECT_ROOT / "data" / "processed" / "fakenewsnet"
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hiperparâmetros ────────────────────────────────────────────────────────
CONFIG = {
    "model_name"    : "bert-base-uncased",
    "max_length"    : 512,
    "batch_size"    : 16,
    "epochs"        : 3,
    "learning_rate" : 2e-5,
    "patience"      : 2,       # early stopping
    "seed"          : 42,
}

# ── Device ─────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Deteta e retorna o melhor dispositivo disponível.

    Prioridade: CUDA (GPU NVIDIA) > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: dispositivo selecionado
    """

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")
    return device

def set_seed(seed: int)-> None:
    """
    Define a seed para reprodutibilidade em todos os geradores aleatórios.

    Args:
        seed: valor da seed (recomendado: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Classes ──────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    """Dataset PyTorch para classificação de fake news."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.encodings["input_ids"][idx],
            "attention_mask" : self.encodings["attention_mask"][idx],
            "labels"         : self.labels[idx],
        }

# ── Mapeamento LIAR 6 classes → binário ────────────────────────────────────
LIAR_LABEL_MAP = {
    "pants-fire" : 0,
    "false"      : 0,
    "barely-true": 0,
    "half-true"  : 1,
    "mostly-true": 1,
    "true"       : 1,
}

def load_liar(split: str, test_run: bool = False) -> pd.DataFrame:
    """
    Carrega o LIAR dataset (TSV raw) e binariza os labels.

    Args:
        split: 'train', 'valid' ou 'test'
        test_run: se True, usa apenas 10% dos dados

    Returns:
        DataFrame com colunas ['text', 'label']
    """
    fname = "valid.tsv" if split == "val" else f"{split}.tsv"
    path  = LIAR_DIR / fname

    # LIAR não tem header — colunas segundo documentação oficial
    cols = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party", "barely_true_c", "false_c",
        "half_true_c", "mostly_true_c", "pants_fire_c", "context"
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)

    # Binarizar label
    df["label"] = df["label"].map(LIAR_LABEL_MAP)
    df = df.dropna(subset=["label", "statement"])
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"statement": "text"})

    if test_run:
        df = df.sample(frac=0.1, random_state=42)
        logger.info(f"LIAR [{split}] test_run — {len(df)} exemplos")
    else:
        logger.info(f"LIAR [{split}] — {len(df)} exemplos")

    return df[["text", "label"]]

# ── Carregar Fakenewsnet ────────────────────────────────────

def load_fakenewsnet(split: str, test_run: bool = False) -> pd.DataFrame:
    """
    Carrega o FakeNewsNet já processado.

    Args:
        split: 'train', 'val' ou 'test'
        test_run: se True, usa apenas 10% dos dados

    Returns:
        DataFrame com colunas ['text', 'label']
    """
    path = FKN_DIR / f"{split}.csv"
    df   = pd.read_csv(path).dropna(subset=["input_text", "label"])
    df   = df.rename(columns={"input_text": "text"})

    if test_run:
        df = df.sample(frac=0.1, random_state=42, replace=True)
        logger.info(f"FakeNewsNet [{split}] test_run — {len(df)} exemplos")
    else:
        logger.info(f"FakeNewsNet [{split}] — {len(df)} exemplos")

    return df[["text", "label"]]

# ── Early Stopping ────────────────────────────────────
class EarlyStopping:
    """Para o treino se a loss de validação não melhorar."""

    def __init__(self, patience: int, model_path: Path):
        self.patience   = patience
        self.model_path = model_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stopped    = False

    def step(self, val_loss: float, model, tokenizer) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)
            logger.info(f" Modelo guardado (val_loss={val_loss:.4f})")
            return False
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stopped = True
                return True
            return False

# ── Loop de treino  ────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, epoch: int, total_epochs: int)-> float:
    """
    Executa uma epoch de treino completa.

    Inclui forward pass, cálculo de loss, backpropagation e atualização
    dos pesos. Aplica gradient clipping (max_norm=1.0) para estabilidade.

    Args:
        model: modelo BERT para classificação
        loader: DataLoader com os dados de treino
        optimizer: otimizador AdamW
        scheduler: scheduler de learning rate com warmup linear
        device: dispositivo de computação (cuda/mps/cpu)
        epoch: época atual (para o tqdm)
        total_epochs: total de épocas (para o tqdm)

    Returns:
        float: loss média da epoch
    """

    model.train()
    total_loss = 0
    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def eval_epoch(model, loader, device) -> tuple:
    """
    Executa uma epoch de avaliação (sem backpropagation).

    Args:
        model: modelo BERT para classificação
        loader: DataLoader com os dados de validação ou teste
        device: dispositivo de computação

    Returns:
        tuple: (avg_loss, predictions, true_labels)
            - avg_loss (float): loss média da epoch
            - predictions (list): predições do modelo (0 ou 1)
            - true_labels (list): labels verdadeiros
    """

    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels

# ── Função principal de treino  ────────────────────────────────────
def run_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_save_path: Path,
    base_model: str,
    config: dict,
    device,
    phase_name: str,
):
    """
    Treina o modelo BERT numa fase (LIAR ou FakeNewsNet).

    Args:
        train_df: dados de treino com colunas ['text', 'label']
        val_df: dados de validação
        model_save_path: onde guardar o melhor modelo
        base_model: nome ou path do modelo base (HuggingFace ou local)
        config: hiperparâmetros
        device: torch device
        phase_name: nome da fase para logging

    Returns:
        histórico de loss {'train': [...], 'val': [...]}
    """
    logger.info(f"=== FASE: {phase_name} ===")
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    ).to(device)

    train_dataset = NewsDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        config["max_length"]
    )
    val_dataset = NewsDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        config["max_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(
        patience=config["patience"],
        model_path=model_save_path
    )

    history = {"train": [], "val": []}

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config["epochs"])
        val_loss, _, _ = eval_epoch(model, val_loader, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        logger.info(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        print(
            f"  Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if early_stopping.step(val_loss, model, tokenizer):
            logger.info("Early stopping ativado!")
            print(" Early stopping ativado!")
            break

    return history

# ── MAIN  ────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning BERT para fake news")
    parser.add_argument(
        "--phase", choices=["1", "2", "all"], default="all",
        help="Fase 1=LIAR, 2=FakeNewsNet, all=ambas"
    )
    parser.add_argument(
        "--test_run", action="store_true",
        help="Usar 10%% dos dados para testar o pipeline"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override ao learning rate (ex: 3e-5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override ao batch size (ex: 8)"
    )
    args = parser.parse_args()

    if args.lr:
        CONFIG["learning_rate"] = args.lr
        logger.info(f"Learning rate override: {args.lr}")

    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
        logger.info(f"Batch size override: {args.batch_size}")

    set_seed(CONFIG["seed"])
    device = get_device()

    all_history = {}

    # ── FASE 1 — LIAR ──────────────────────────────────────────────────────
    if args.phase in ["1", "all"]:
        print("\n FASE 1 — Pré-fine-tune no LIAR")
        train_df = load_liar("train", test_run=args.test_run)
        val_df   = load_liar("val",   test_run=args.test_run)

        save_path = MODELS_DIR / "bert_liar"
        history   = run_training(
            train_df, val_df,
            model_save_path=save_path,
            base_model=CONFIG["model_name"],
            config=CONFIG,
            device=device,
            phase_name="LIAR"
        )
        all_history["phase1_liar"] = history
        print(f" Fase 1 concluída! Modelo em: {save_path}")

    # ── FASE 2 — FakeNewsNet ───────────────────────────────────────────────
    if args.phase in ["2", "all"]:
        print("\n FASE 2 — Fine-tune no FakeNewsNet")

        # Usar modelo da Fase 1 como base (se existir)
        liar_model_path = MODELS_DIR / "bert_liar"
        base_model = str(liar_model_path) if liar_model_path.exists() else CONFIG["model_name"]
        logger.info(f"Base model para Fase 2: {base_model}")

        train_df = load_fakenewsnet("train", test_run=args.test_run)
        val_df   = load_fakenewsnet("val",   test_run=args.test_run)

        save_path = MODELS_DIR / "bert_fake_news"
        history   = run_training(
            train_df, val_df,
            model_save_path=save_path,
            base_model=base_model,
            config=CONFIG,
            device=device,
            phase_name="FakeNewsNet"
        )
        all_history["phase2_fakenewsnet"] = history
        print(f" Fase 2 concluída! Modelo em: {save_path}")

    # ── Guardar histórico ──────────────────────────────────────────────────
    history_path = RESULTS_DIR / "bert_training_history.json"
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2)
    logger.info(f"Histórico guardado em {history_path}")
    print(f"\n Histórico guardado em {history_path}")


if __name__ == "__main__":
    main()