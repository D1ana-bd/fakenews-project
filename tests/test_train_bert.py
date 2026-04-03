"""
test_train_bert.py
─────────────────────────────────────────────────────────────────────────────
Testes unitários para o módulo train_bert.py.
Cobre: get_device, set_seed, load_liar, load_fakenewsnet,
       NewsDataset, EarlyStopping.

Autora : Diana Dória
Projeto: Sistema Inteligente de Deteção e Mitigação de Desinformação
         em Redes Sociais | UCP 2025/26
─────────────────────────────────────────────────────────────────────────────

Como correr:
    pytest tests/test_train_bert.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.FKNWS.models.train_bert import (
    get_device,
    set_seed,
    load_liar,
    load_fakenewsnet,
    NewsDataset,
    EarlyStopping,
    LIAR_LABEL_MAP,
    CONFIG,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_liar_tsv(tmp_path):
    """Cria um ficheiro TSV dummy com estrutura do LIAR."""
    content = (
        "id1\ttrue\tThis is a real statement\tpolitics\tJohn\tjob\tstate\tdem\t1\t2\t3\t4\t5\tcontext\n"
        "id2\tfalse\tThis is a fake statement\thealth\tJane\tjob\tstate\trep\t1\t2\t3\t4\t5\tcontext\n"
        "id3\tpants-fire\tExtreme fake statement\teconomy\tBob\tjob\tstate\tdem\t1\t2\t3\t4\t5\tcontext\n"
        "id4\tmostly-true\tMostly real statement\tforeign\tAlice\tjob\tstate\trep\t1\t2\t3\t4\t5\tcontext\n"
        "id5\tbarely-true\tBarely true statement\tsci\tEve\tjob\tstate\tdem\t1\t2\t3\t4\t5\tcontext\n"
        "id6\thalf-true\tHalf true statement\tlaw\tMike\tjob\tstate\trep\t1\t2\t3\t4\t5\tcontext\n"
    )
    tsv_file = tmp_path / "train.tsv"
    tsv_file.write_text(content)
    return tmp_path


@pytest.fixture
def dummy_fakenewsnet_csv(tmp_path):
    """Cria um CSV dummy com estrutura do FakeNewsNet processado."""
    df = pd.DataFrame({
        "id"          : ["a1", "a2", "a3", "a4"],
        "title_clean" : ["title1", "title2", "title3", "title4"],
        "text_clean"  : ["body1", "body2", "body3", "body4"],
        "input_text"  : ["text one", "text two", "text three", "text four"],
        "publish_date": ["2020-01-01"] * 4,
        "label"       : [0, 1, 0, 1],
        "source"      : ["gossipcop"] * 4,
    })
    csv_file = tmp_path / "train.csv"
    df.to_csv(csv_file, index=False)
    return tmp_path


@pytest.fixture
def dummy_tokenizer():
    """Mock de tokenizer HuggingFace."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids"      : torch.zeros(4, 512, dtype=torch.long),
        "attention_mask" : torch.ones(4, 512, dtype=torch.long),
    }
    return tokenizer


# ── Testes: get_device ────────────────────────────────────────────────────────

class TestGetDevice:

    def test_returns_torch_device(self):
        """get_device deve sempre retornar um torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cuda_prioritized(self):
        """Se CUDA disponível, deve retornar cuda."""
        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
            assert device.type == "cuda"

    def test_mps_prioritized_over_cpu(self):
        """Se MPS disponível e CUDA não, deve retornar mps."""
        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.cuda.is_available", return_value=False):
            device = get_device()
            assert device.type == "mps"

    def test_fallback_cpu(self):
        """Se nem CUDA nem MPS, deve retornar cpu."""
        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.cuda.is_available", return_value=False):
            device = get_device()
            assert device.type == "cpu"


# ── Testes: set_seed ──────────────────────────────────────────────────────────

class TestSetSeed:

    def test_reproducibility(self):
        """Mesma seed deve produzir mesmos valores aleatórios."""
        set_seed(42)
        val1 = torch.rand(1).item()
        set_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    def test_different_seeds_differ(self):
        """Seeds diferentes devem (quase sempre) produzir valores diferentes."""
        set_seed(42)
        val1 = torch.rand(10)
        set_seed(99)
        val2 = torch.rand(10)
        assert not torch.allclose(val1, val2)


# ── Testes: LIAR_LABEL_MAP ────────────────────────────────────────────────────

class TestLiarLabelMap:

    def test_fake_labels_are_zero(self):
        """Labels de desinformação devem mapear para 0."""
        for label in ["pants-fire", "false", "barely-true"]:
            assert LIAR_LABEL_MAP[label] == 0, f"{label} devia ser 0"

    def test_real_labels_are_one(self):
        """Labels de informação real devem mapear para 1."""
        for label in ["half-true", "mostly-true", "true"]:
            assert LIAR_LABEL_MAP[label] == 1, f"{label} devia ser 1"

    def test_all_six_classes_covered(self):
        """Todos os 6 labels originais do LIAR devem estar mapeados."""
        assert len(LIAR_LABEL_MAP) == 6


# ── Testes: load_liar ─────────────────────────────────────────────────────────

class TestLoadLiar:

    def test_returns_dataframe(self, dummy_liar_tsv):
        """load_liar deve retornar um DataFrame."""
        with patch("src.FKNWS.models.train_bert.LIAR_DIR", dummy_liar_tsv):
            df = load_liar("train")
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, dummy_liar_tsv):
        """DataFrame deve ter colunas 'text' e 'label'."""
        with patch("src.FKNWS.models.train_bert.LIAR_DIR", dummy_liar_tsv):
            df = load_liar("train")
        assert "text" in df.columns
        assert "label" in df.columns

    def test_labels_are_binary(self, dummy_liar_tsv):
        """Labels devem ser apenas 0 ou 1."""
        with patch("src.FKNWS.models.train_bert.LIAR_DIR", dummy_liar_tsv):
            df = load_liar("train")
        assert set(df["label"].unique()).issubset({0, 1})

    def test_test_run_reduces_size(self, dummy_liar_tsv):
        """test_run=True deve retornar menos exemplos."""
        with patch("src.FKNWS.models.train_bert.LIAR_DIR", dummy_liar_tsv):
            df_full     = load_liar("train", test_run=False)
            df_test_run = load_liar("train", test_run=True)
        assert len(df_test_run) <= len(df_full)

    def test_no_nulls_in_output(self, dummy_liar_tsv):
        """Não devem existir nulos em text ou label."""
        with patch("src.FKNWS.models.train_bert.LIAR_DIR", dummy_liar_tsv):
            df = load_liar("train")
        assert df["text"].isnull().sum() == 0
        assert df["label"].isnull().sum() == 0


# ── Testes: load_fakenewsnet ──────────────────────────────────────────────────

class TestLoadFakenewsnet:

    def test_returns_dataframe(self, dummy_fakenewsnet_csv):
        """load_fakenewsnet deve retornar um DataFrame."""
        with patch("src.FKNWS.models.train_bert.FKN_DIR", dummy_fakenewsnet_csv):
            df = load_fakenewsnet("train")
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, dummy_fakenewsnet_csv):
        """DataFrame deve ter colunas 'text' e 'label'."""
        with patch("src.FKNWS.models.train_bert.FKN_DIR", dummy_fakenewsnet_csv):
            df = load_fakenewsnet("train")
        assert "text" in df.columns
        assert "label" in df.columns

    def test_labels_are_binary(self, dummy_fakenewsnet_csv):
        """Labels devem ser apenas 0 ou 1."""
        with patch("src.FKNWS.models.train_bert.FKN_DIR", dummy_fakenewsnet_csv):
            df = load_fakenewsnet("train")
        assert set(df["label"].unique()).issubset({0, 1})

    def test_no_nulls_in_output(self, dummy_fakenewsnet_csv):
        """Não devem existir nulos em text ou label."""
        with patch("src.FKNWS.models.train_bert.FKN_DIR", dummy_fakenewsnet_csv):
            df = load_fakenewsnet("train")
        assert df["text"].isnull().sum() == 0
        assert df["label"].isnull().sum() == 0


# ── Testes: NewsDataset ───────────────────────────────────────────────────────

class TestNewsDataset:

    def test_len(self, dummy_tokenizer):
        """__len__ deve retornar o número de exemplos."""
        texts  = ["text one", "text two", "text three"]
        labels = [0, 1, 0]
        with patch("src.FKNWS.models.train_bert.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.return_value = dummy_tokenizer
            dataset = NewsDataset(texts, labels, dummy_tokenizer, max_length=128)
        assert len(dataset) == 3

    def test_getitem_keys(self, dummy_tokenizer):
        """Cada item deve ter input_ids, attention_mask e labels."""
        texts  = ["text one", "text two", "text three", "text four"]
        labels = [0, 1, 0, 1]
        dataset = NewsDataset(texts, labels, dummy_tokenizer, max_length=128)
        item = dataset[0]
        assert "input_ids"       in item
        assert "attention_mask"  in item
        assert "labels"          in item

    def test_labels_are_tensors(self, dummy_tokenizer):
        """Labels devem ser tensores torch.long."""
        texts  = ["text one", "text two", "text three", "text four"]
        labels = [0, 1, 0, 1]
        dataset = NewsDataset(texts, labels, dummy_tokenizer, max_length=128)
        assert dataset[0]["labels"].dtype == torch.long


# ── Testes: EarlyStopping ─────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_saves_on_improvement(self, tmp_path):
        """Deve guardar modelo quando val_loss melhora."""
        mock_model     = MagicMock()
        mock_tokenizer = MagicMock()
        es = EarlyStopping(patience=2, model_path=tmp_path / "model")
        stopped = es.step(0.5, mock_model, mock_tokenizer)
        assert not stopped
        mock_model.save_pretrained.assert_called_once()

    def test_counter_increments_on_no_improvement(self, tmp_path):
        """Counter deve incrementar quando val_loss não melhora."""
        mock_model     = MagicMock()
        mock_tokenizer = MagicMock()
        es = EarlyStopping(patience=2, model_path=tmp_path / "model")
        es.step(0.5, mock_model, mock_tokenizer)   # melhora
        es.step(0.6, mock_model, mock_tokenizer)   # não melhora
        assert es.counter == 1

    def test_stops_after_patience(self, tmp_path):
        """Deve retornar True após patience epochs sem melhoria."""
        mock_model     = MagicMock()
        mock_tokenizer = MagicMock()
        es = EarlyStopping(patience=2, model_path=tmp_path / "model")
        es.step(0.5, mock_model, mock_tokenizer)   # melhora
        es.step(0.6, mock_model, mock_tokenizer)   # não melhora (counter=1)
        stopped = es.step(0.7, mock_model, mock_tokenizer)  # não melhora (counter=2)
        assert stopped

    def test_resets_counter_on_improvement(self, tmp_path):
        """Counter deve resetar quando val_loss melhora novamente."""
        mock_model     = MagicMock()
        mock_tokenizer = MagicMock()
        es = EarlyStopping(patience=3, model_path=tmp_path / "model")
        es.step(0.5, mock_model, mock_tokenizer)   # melhora
        es.step(0.6, mock_model, mock_tokenizer)   # não melhora (counter=1)
        es.step(0.4, mock_model, mock_tokenizer)   # melhora! counter reset
        assert es.counter == 0

    def test_best_loss_updates(self, tmp_path):
        """best_loss deve atualizar quando val_loss melhora."""
        mock_model     = MagicMock()
        mock_tokenizer = MagicMock()
        es = EarlyStopping(patience=2, model_path=tmp_path / "model")
        es.step(0.5, mock_model, mock_tokenizer)
        assert es.best_loss == 0.5
        es.step(0.3, mock_model, mock_tokenizer)
        assert es.best_loss == 0.3


# ── Testes: CONFIG ────────────────────────────────────────────────────────────

class TestConfig:

    def test_required_keys_exist(self):
        """CONFIG deve ter todos os hiperparâmetros necessários."""
        required = ["model_name", "max_length", "batch_size", "epochs",
                    "learning_rate", "patience", "seed"]
        for key in required:
            assert key in CONFIG, f"CONFIG está a faltar: {key}"

    def test_sensible_values(self):
        """Hiperparâmetros devem ter valores razoáveis."""
        assert CONFIG["batch_size"] > 0
        assert CONFIG["epochs"] > 0
        assert 0 < CONFIG["learning_rate"] < 1
        assert CONFIG["patience"] > 0