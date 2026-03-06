"""
test_data_preprocessing.py
─────────────────────────────────────────────────────────────
Testes unitários para o módulo data_preprocessing.py.
Cobre: clean_text, parse_publish_date, load_fakenewsnet,
       e preprocess_pipeline.

Autora : Diana Dória
Projeto: Sistema Inteligente de Deteção e Mitigação de Desinformação em Redes Sociais | UCP 2025/26
─────────────────────────────────────────────────────────────

Como correr:
    pytest tests/test_data_preprocessing.py -v
─────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import tempfile
import os

# Adicionar src ao path para importar o módulo
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from src.FKNWS.data_prep.data_preprocessing import (
    clean_text,
    parse_publish_date,
    load_fakenewsnet,
    preprocess_pipeline,
)


# ═════════════════════════════════════════════════════════════
# FIXTURES — dados reutilizáveis nos testes
# ═════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df_fake():
    """DataFrame simulado de artigos fake BuzzFeed — IDs únicos globalmente."""
    return pd.DataFrame({
        "id":           ["buzz_fake_001", "buzz_fake_002", "buzz_fake_003"],
        "title":        [
            "BREAKING: Shocking news about vaccines!!!",
            "You won't believe what happened next... http://spam.com",
            "Secret government plan EXPOSED",
        ],
        "text":         [
            "This is a fake article with a URL https://fake.com and an email test@fake.com.",
            "Another fake article with special chars and noise to clean up properly.",
            "Short fake text that is long enough to pass the minimum length filter.",
        ],
        "publish_date": [
            "{'$date': 1474243200000}",
            "{'$date': 1474416521000}",
            None,
        ],
    })


@pytest.fixture
def sample_df_real():
    """DataFrame simulado de artigos reais BuzzFeed — IDs únicos globalmente."""
    return pd.DataFrame({
        "id":           ["buzz_real_001", "buzz_real_002", "buzz_real_003"],
        "title":        [
            "Scientists publish new climate study",
            "Election results confirmed by officials",
            "Local hospital opens new wing",
        ],
        "text":         [
            "A peer-reviewed study published in Nature found that temperatures are rising.",
            "Officials confirmed the election results after a thorough recount process.",
            "The new hospital wing will serve over 10,000 patients annually.",
        ],
        "publish_date": [
            "{'$date': 1474243200000}",
            "2022-05-10T12:00:00",
            None,
        ],
    })


@pytest.fixture
def sample_df_fake_politi():
    """DataFrame simulado de artigos fake PolitiFact — IDs únicos globalmente."""
    return pd.DataFrame({
        "id":           ["poli_fake_001", "poli_fake_002", "poli_fake_003"],
        "title":        [
            "Politician claims false statistics in speech",
            "Viral post about economy is completely wrong",
            "Misleading headline about immigration numbers",
        ],
        "text":         [
            "A politician made several false claims during a recent speech about the economy.",
            "A viral post spreading misinformation about economic data has been debunked.",
            "Reports about immigration numbers contain several misleading statistics.",
        ],
        "publish_date": [
            "{'$date': 1474243200000}",
            "{'$date': 1474416521000}",
            None,
        ],
    })


@pytest.fixture
def sample_df_real_politi():
    """DataFrame simulado de artigos reais PolitiFact — IDs únicos globalmente."""
    return pd.DataFrame({
        "id":           ["poli_real_001", "poli_real_002", "poli_real_003"],
        "title":        [
            "New climate report published by scientists",
            "Government announces new infrastructure plan",
            "Health officials confirm vaccination rates rising",
        ],
        "text":         [
            "Scientists have published a new peer-reviewed report on climate change impacts.",
            "The government has announced a major new infrastructure investment plan.",
            "Health officials confirmed that vaccination rates have increased significantly.",
        ],
        "publish_date": [
            "{'$date': 1474243200000}",
            "2022-05-10T12:00:00",
            None,
        ],
    })


# ═════════════════════════════════════════════════════════════
# TESTES: clean_text
# ═════════════════════════════════════════════════════════════

class TestCleanText:

    def test_remove_url(self):
        """URLs devem ser removidos do texto."""
        text = "Check this out https://fake-news.com for more info"
        result = clean_text(text)
        assert "http" not in result
        assert "fake-news.com" not in result

    def test_remove_www_url(self):
        """URLs com www devem ser removidos."""
        text = "Visit www.example.com today"
        result = clean_text(text)
        assert "www" not in result

    def test_remove_email(self):
        """Emails devem ser removidos do texto."""
        text = "Contact us at info@fakenews.org for details"
        result = clean_text(text)
        assert "@" not in result

    def test_remove_emojis(self):
        """Emojis e caracteres não-ASCII devem ser removidos."""
        text = "Breaking news 🚨🔥 you won't believe this 😂"
        result = clean_text(text)
        assert "🚨" not in result
        assert "🔥" not in result
        assert "😂" not in result

    def test_normalize_whitespace(self):
        """Múltiplos espaços devem ser normalizados para um só."""
        text = "This   has   too    many    spaces"
        result = clean_text(text)
        assert "  " not in result

    def test_strip_whitespace(self):
        """Espaços no início e fim devem ser removidos."""
        text = "   text with spaces around   "
        result = clean_text(text)
        assert result == result.strip()

    def test_nan_input(self):
        """NaN deve retornar string vazia."""
        result = clean_text(float("nan"))
        assert isinstance(result, str)
        assert result == ""

    def test_none_input(self):
        """None deve ser tratado sem erro."""
        result = clean_text(None)
        assert isinstance(result, str)

    def test_empty_string(self):
        """String vazia deve retornar string vazia."""
        result = clean_text("")
        assert result == ""

    def test_normal_text_preserved(self):
        """Texto normal sem ruído deve ser preservado."""
        text = "Scientists found new evidence about climate change."
        result = clean_text(text)
        assert "Scientists" in result
        assert "climate change" in result

    def test_returns_string(self):
        """A função deve sempre retornar uma string."""
        for input_val in ["hello", 123, None, float("nan"), ""]:
            result = clean_text(input_val)
            assert isinstance(result, str)

    def test_combined_noise(self):
        """Texto com múltiplos tipos de ruído deve ser limpo corretamente."""
        text = "  BREAKING 🔥 visit https://spam.com or email us@spam.com   "
        result = clean_text(text)
        assert "http" not in result
        assert "@" not in result
        assert "🔥" not in result
        assert result == result.strip()


# ═════════════════════════════════════════════════════════════
# TESTES: parse_publish_date
# ═════════════════════════════════════════════════════════════

class TestParsePublishDate:

    def test_json_timestamp_format(self):
        """Formato {'$date': timestamp} deve ser convertido correctamente."""
        date_val = "{'$date': 1474243200000}"
        result = parse_publish_date(date_val)
        assert result == "2016-09-19"

    def test_iso_string_format(self):
        """String ISO de data deve retornar os primeiros 10 caracteres."""
        date_val = "2022-05-10T12:00:00"
        result = parse_publish_date(date_val)
        assert result == "2022-05-10"

    def test_none_returns_unknown(self):
        """None deve retornar 'unknown'."""
        result = parse_publish_date(None)
        assert result == "unknown"

    def test_nan_returns_unknown(self):
        """NaN deve retornar 'unknown'."""
        result = parse_publish_date(float("nan"))
        assert result == "unknown"

    def test_invalid_string_returns_unknown(self):
        """String inválida deve retornar 'unknown'."""
        result = parse_publish_date("not a date at all")
        assert result == "unknown"

    def test_returns_string(self):
        """A função deve sempre retornar uma string."""
        for input_val in ["{'$date': 1474243200000}", None, float("nan"), "2022-01-01"]:
            result = parse_publish_date(input_val)
            assert isinstance(result, str)

    def test_output_format(self):
        """O resultado deve ter formato YYYY-MM-DD ou 'unknown'."""
        import re
        date_val = "{'$date': 1474243200000}"
        result = parse_publish_date(date_val)
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        assert re.match(pattern, result), f"Formato inesperado: {result}"


# ═════════════════════════════════════════════════════════════
# TESTES: load_fakenewsnet
# ═════════════════════════════════════════════════════════════

class TestLoadFakenewsnet:

    def test_load_with_mock_files(self, tmp_path, sample_df_fake, sample_df_real,
                                   sample_df_fake_politi, sample_df_real_politi):
        """Deve carregar e combinar ficheiros correctamente."""
        sample_df_fake[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_fake_news_content.csv", index=False)
        sample_df_real[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_real_news_content.csv", index=False)
        sample_df_fake_politi[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "PolitiFact_fake_news_content.csv", index=False)
        sample_df_real_politi[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "PolitiFact_real_news_content.csv", index=False)

        result = load_fakenewsnet(raw_dir=tmp_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12  # 3 fake + 3 real × 2 fontes
        assert "label" in result.columns
        assert "source" in result.columns

    def test_labels_correct(self, tmp_path, sample_df_fake, sample_df_real,
                             sample_df_fake_politi, sample_df_real_politi):
        """Labels 0 e 1 devem ser atribuídas correctamente."""
        sample_df_fake[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_fake_news_content.csv", index=False)
        sample_df_real[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_real_news_content.csv", index=False)
        sample_df_fake_politi[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "PolitiFact_fake_news_content.csv", index=False)
        sample_df_real_politi[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "PolitiFact_real_news_content.csv", index=False)

        result = load_fakenewsnet(raw_dir=tmp_path)

        assert set(result["label"].unique()) == {0, 1}

    def test_missing_file_warning(self, tmp_path, sample_df_fake, caplog):
        """Ficheiro em falta deve gerar warning sem crashar."""
        sample_df_fake[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_fake_news_content.csv", index=False)

        try:
            result = load_fakenewsnet(raw_dir=tmp_path)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"load_fakenewsnet lançou excepção inesperada: {e}")

    def test_source_column_values(self, tmp_path, sample_df_fake, sample_df_real,
                                   sample_df_fake_politi, sample_df_real_politi):
        """Coluna source deve conter apenas 'buzzfeed' ou 'politifact'."""
        sample_df_fake[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_fake_news_content.csv", index=False)
        sample_df_real[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "BuzzFeed_real_news_content.csv", index=False)
        sample_df_fake_politi[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "PolitiFact_fake_news_content.csv", index=False)
        sample_df_real_politi[["id", "title", "text", "publish_date"]].to_csv(
            tmp_path / "PolitiFact_real_news_content.csv", index=False)

        result = load_fakenewsnet(raw_dir=tmp_path)

        assert set(result["source"].unique()).issubset({"buzzfeed", "politifact"})


# ═════════════════════════════════════════════════════════════
# TESTES: preprocess_pipeline
# ═════════════════════════════════════════════════════════════

class TestPreprocessPipeline:

    @pytest.fixture
    def setup_raw_dir(self, tmp_path, sample_df_fake, sample_df_real,
                      sample_df_fake_politi, sample_df_real_politi):
        """Cria estrutura de pastas temporária com CSVs de teste — IDs únicos."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        sample_df_fake[["id", "title", "text", "publish_date"]].to_csv(
            raw_dir / "BuzzFeed_fake_news_content.csv", index=False)
        sample_df_real[["id", "title", "text", "publish_date"]].to_csv(
            raw_dir / "BuzzFeed_real_news_content.csv", index=False)
        sample_df_fake_politi[["id", "title", "text", "publish_date"]].to_csv(
            raw_dir / "PolitiFact_fake_news_content.csv", index=False)
        sample_df_real_politi[["id", "title", "text", "publish_date"]].to_csv(
            raw_dir / "PolitiFact_real_news_content.csv", index=False)

        return raw_dir, processed_dir

    def test_returns_dict_with_three_splits(self, setup_raw_dir):
        """Pipeline deve retornar dict com train, val e test."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"train", "val", "test"}

    def test_splits_are_dataframes(self, setup_raw_dir):
        """Cada split deve ser um DataFrame."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        for split_name in ["train", "val", "test"]:
            assert isinstance(result[split_name], pd.DataFrame), \
                f"Split '{split_name}' não é um DataFrame"

    def test_no_data_leakage(self, setup_raw_dir):
        """Não deve haver sobreposição de IDs entre splits."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        train_ids = set(result["train"]["id"])
        val_ids   = set(result["val"]["id"])
        test_ids  = set(result["test"]["id"])

        assert train_ids.isdisjoint(val_ids),  "Sobreposição entre train e val!"
        assert train_ids.isdisjoint(test_ids), "Sobreposição entre train e test!"
        assert val_ids.isdisjoint(test_ids),   "Sobreposição entre val e test!"

    def test_total_rows_preserved(self, setup_raw_dir):
        """A soma dos splits deve ser igual ao total original."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        total_splits = sum(len(result[k]) for k in ["train", "val", "test"])
        # Total original = 12 (3 fake + 3 real) × 2 fontes
        # Pode ser ligeiramente menor se linhas forem removidas na limpeza
        assert total_splits <= 12
        assert total_splits > 0

    def test_input_text_column_exists(self, setup_raw_dir):
        """Coluna input_text deve existir em todos os splits."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        for split_name, df in result.items():
            assert "input_text" in df.columns, \
                f"Coluna 'input_text' em falta no split '{split_name}'"

    def test_input_text_contains_sep(self, setup_raw_dir):
        """input_text deve conter o token [SEP]."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        for split_name, df in result.items():
            assert df["input_text"].str.contains(r"\[SEP\]").all(), \
                f"Token [SEP] em falta no split '{split_name}'"

    def test_label_column_binary(self, setup_raw_dir):
        """Labels devem ser apenas 0 ou 1."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        for split_name, df in result.items():
            assert set(df["label"].unique()).issubset({0, 1}), \
                f"Labels inválidas no split '{split_name}'"

    def test_csv_files_created(self, setup_raw_dir):
        """Ficheiros train.csv, val.csv, test.csv devem ser criados."""
        raw_dir, processed_dir = setup_raw_dir
        preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        for fname in ["train.csv", "val.csv", "test.csv"]:
            fpath = processed_dir / fname
            assert fpath.exists(), f"Ficheiro {fname} não foi criado!"
            assert fpath.stat().st_size > 0, f"Ficheiro {fname} está vazio!"

    def test_reproducibility(self, setup_raw_dir):
        """Pipeline com mesma seed deve produzir resultados idênticos."""
        raw_dir, processed_dir = setup_raw_dir

        result1 = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir,
                                      random_state=42)
        result2 = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir,
                                      random_state=42)

        for split_name in ["train", "val", "test"]:
            pd.testing.assert_frame_equal(
                result1[split_name].reset_index(drop=True),
                result2[split_name].reset_index(drop=True),
                check_like=True
            )

    def test_clean_text_applied(self, setup_raw_dir):
        """Colunas title_clean e text_clean não devem conter URLs."""
        raw_dir, processed_dir = setup_raw_dir
        result = preprocess_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)

        all_data = pd.concat(result.values())
        assert not all_data["title_clean"].str.contains("http", na=False).any(), \
            "URLs encontrados em title_clean!"
        assert not all_data["text_clean"].str.contains("http", na=False).any(), \
            "URLs encontrados em text_clean!"