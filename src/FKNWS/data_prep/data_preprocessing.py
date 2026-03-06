"""
data_preprocessing.py
─────────────────────────────────────────────────────────────
Funções de pré-processamento para o dataset FakeNewsNet.
Pipeline: limpeza de texto → criação de input BERT → splits

Autora : Diana Dória
Projeto: Sistema Inteligente de Deteção e Mitigação de Desinformação em Redes Sociais| UCP 2025/26
─────────────────────────────────────────────────────────────
"""

import re
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# Logger
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.FKNWS.utils.get_logger import get_logger

logger = get_logger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw/fakenewsnet")
PROCESSED_DIR = Path("data/processed/fakenewsnet")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Limpeza de texto ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpa um texto para input no BERT.

    Operações:
        1. Converte para string (caso seja NaN ou outro tipo)
        2. Remove URLs
        3. Remove emails
        4. Remove emojis e caracteres não-ASCII
        5. Remove caracteres especiais (mantém pontuação básica)
        6. Normaliza espaços em branco
        7. Strip final

    Args:
        text: texto bruto

    Returns:
        texto limpo como string
    """
    # 1. Garantir que é string
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""

    # 2. Remover URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remover emails
    text = re.sub(r"\S+@\S+", "", text)

    # 4. Remover emojis e caracteres não-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")

    # 5. Remover caracteres especiais (manter letras, números, pontuação básica)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:\'\"-]", " ", text)

    # 6. Normalizar espaços (múltiplos espaços → um)
    text = re.sub(r"\s+", " ", text)

    # 7. Strip
    return text.strip()


# ── Conversão de datas ────────────────────────────────────────────────────────

def parse_publish_date(date_val) -> str:
    """
    Converte publish_date do formato {'$date': timestamp_ms} para string legível.

    Args:
        date_val: valor da coluna publish_date (string, dict, ou NaN)

    Returns:
        string no formato 'YYYY-MM-DD' ou 'unknown'
    """
    try:
        if pd.isna(date_val):
            return "unknown"
    except Exception:
        pass

    try:
        # Formato: "{'$date': 1474243200000}"
        if isinstance(date_val, str) and "$date" in date_val:
            d = json.loads(date_val.replace("'", '"'))
            timestamp_ms = d["$date"]
            return datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")

        # Já é string de data normal — validar formato YYYY-MM-DD
        if isinstance(date_val, str):
            candidate = date_val[:10]
            # Verificar que tem formato de data válido
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate

    except Exception:
        return "unknown"

    return "unknown"


# ── Carregamento dos dados ────────────────────────────────────────────────────

def load_fakenewsnet(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Carrega e combina os ficheiros de conteúdo do FakeNewsNet.
    Usa apenas os ficheiros com texto completo (BuzzFeed + PolitiFact content).

    Returns:
        DataFrame com colunas: id, title, text, publish_date, label, source
    """
    file_map = {
        "buzzfeed_fake":   ("BuzzFeed_fake_news_content.csv",  1),
        "buzzfeed_real":   ("BuzzFeed_real_news_content.csv",  0),
        "politifact_fake": ("PolitiFact_fake_news_content.csv", 1),
        "politifact_real": ("PolitiFact_real_news_content.csv", 0),
    }

    dfs = []
    for source_name, (fname, label) in file_map.items():
        fpath = raw_dir / fname
        if not fpath.exists():
            logger.warning(f"Ficheiro não encontrado: {fname}")
            continue

        df = pd.read_csv(fpath)

        # Manter só colunas relevantes
        cols_keep = [c for c in ["id", "title", "text", "publish_date"] if c in df.columns]
        df = df[cols_keep].copy()

        df["label"]  = label
        df["source"] = source_name.split("_")[0]  # buzzfeed / politifact
        dfs.append(df)
        logger.info(f"Carregado: {fname} → {len(df)} artigos [label={label}]")

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total carregado: {len(df_all)} artigos")
    return df_all


# ── Pipeline principal ────────────────────────────────────────────────────────

def preprocess_pipeline(raw_dir: Path = RAW_DIR,
                        processed_dir: Path = PROCESSED_DIR,
                        test_size: float = 0.10,
                        val_size: float = 0.10,
                        random_state: int = 42) -> dict:
    """
    Pipeline completo de pré-processamento:
        1. Carrega dados brutos
        2. Limpa título e texto
        3. Converte datas
        4. Cria coluna input_text = title + [SEP] + text
        5. Cria splits train/val/test estratificados
        6. Guarda CSVs em processed/

    Args:
        raw_dir       : pasta com os CSVs brutos
        processed_dir : pasta de destino dos ficheiros processados
        test_size     : proporção do conjunto de teste (default 10%)
        val_size      : proporção do conjunto de validação (default 10%)
        random_state  : seed para reprodutibilidade

    Returns:
        dict com os três DataFrames: {"train": df, "val": df, "test": df}
    """
    logger.info("=" * 50)
    logger.info("INÍCIO DO PIPELINE DE PRÉ-PROCESSAMENTO")
    logger.info("=" * 50)

    # 1. Carregar
    logger.info("[1/5] A carregar dados...")
    df = load_fakenewsnet(raw_dir)

    # 2. Limpar título e texto
    logger.info("[2/5] A limpar texto...")
    df["title_clean"] = df["title"].apply(clean_text)
    df["text_clean"]  = df["text"].apply(clean_text)

    # Remover linhas com texto vazio após limpeza
    before = len(df)
    df = df[df["text_clean"].str.len() > 10].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.warning(f"{removed} linhas removidas (texto vazio após limpeza)")
    else:
        logger.info("Nenhuma linha removida após limpeza")

    # 3. Converter datas
    logger.info("[3/5] A converter datas...")
    if "publish_date" in df.columns:
        df["publish_date"] = df["publish_date"].apply(parse_publish_date)
    logger.info("Datas convertidas com sucesso")

    # 4. Criar input_text para BERT
    logger.info("[4/5] A criar input_text para BERT...")
    df["input_text"] = df["title_clean"] + " [SEP] " + df["text_clean"]
    logger.info("input_text criado (title + [SEP] + text)")

    # Colunas finais
    cols_final = ["id", "title_clean", "text_clean", "input_text",
                  "publish_date", "label", "source"]
    cols_final = [c for c in cols_final if c in df.columns]
    df = df[cols_final]

    # 5. Criar splits estratificados
    logger.info("[5/5] A criar splits train/val/test...")

    # Primeiro split: separar test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    # Segundo split: separar val do train
    val_size_adjusted = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_adjusted,
        stratify=df_train_val["label"],
        random_state=random_state
    )

    # Guardar CSVs
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(processed_dir / "train.csv", index=False)
    df_val.to_csv(processed_dir   / "val.csv",   index=False)
    df_test.to_csv(processed_dir  / "test.csv",  index=False)

    # Relatório final no terminal
    print(f"\n{'─' * 55}")
    print(f"  {'Split':<10} {'Total':>8} {'Fake':>8} {'Real':>8} {'% Fake':>8}")
    print(f"{'─' * 55}")
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        total = len(split_df)
        fake  = (split_df["label"] == 1).sum()
        real  = (split_df["label"] == 0).sum()
        print(f"  {name:<10} {total:>8} {fake:>8} {real:>8} {fake/total*100:>7.1f}%")
        logger.info(f"Split {name}: {total} artigos | fake={fake} | real={real} | {fake/total*100:.1f}% fake")
    print(f"{'─' * 55}")

    logger.info(f"Ficheiros guardados em: {processed_dir}")
    logger.info("PIPELINE CONCLUÍDO COM SUCESSO")

    return {"train": df_train, "val": df_val, "test": df_test}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    splits = preprocess_pipeline()