"""
test_train_bert_titles.py
=========================
Testes unitários para src/FKNWS/models/train_bert_titles.py

Cobre:
    - Data augmentation (tipos A, B, C)
    - TitlesDataset (PyTorch Dataset)
    - evaluate_on_test (inferência no test set)
    - Estrutura do output JSON
    - Edge cases: textos curtos, textos vazios

Uso:
    pytest tests/test_train_bert_titles.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from FKNWS.models.train_bert_titles import (
    TitlesDataset,
    augment_titles,
    evaluate_on_test,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df_train():
    """DataFrame de treino com artigos de diferentes tamanhos."""
    return pd.DataFrame({
        "id":          ["Fake_1", "Real_1", "Fake_2", "Real_2"],
        "title_clean": [
            "Government hiding secret cure",
            "Federal Reserve raises rates",
            "SHOCKING conspiracy revealed",
            "Scientists publish new research",
        ],
        "text_clean": [
            # texto longo (> 100 words)
            " ".join(["word"] * 150),
            # texto médio (50-100 words)
            " ".join(["word"] * 70),
            # texto curto (10-50 words)
            " ".join(["word"] * 20),
            # texto muito curto (< 10 words)
            "short text",
        ],
        "label": [1, 0, 1, 0],
    })


@pytest.fixture
def sample_df_test():
    """DataFrame de teste com títulos."""
    return pd.DataFrame({
        "id":          ["Fake_1", "Real_1", "Fake_2", "Real_2", "Fake_3"],
        "title_clean": [
            "Government hiding secret cure",
            "Federal Reserve raises rates",
            "SHOCKING conspiracy revealed",
            "Scientists publish new research",
            "You won't believe this",
        ],
        "label": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def mock_tokenizer():
    """Mock do tokenizer que devolve tensores reais."""
    def fake_tokenize(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        return {
            "input_ids":      torch.ones(batch_size, 10, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
        }
    tokenizer = MagicMock(side_effect=fake_tokenize)
    tokenizer.return_value = {
        "input_ids":      torch.ones(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock do modelo que devolve logits reais."""
    def fake_forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        output = MagicMock()
        output.logits = torch.tensor(
            [[0.3, 0.7]] * batch_size, dtype=torch.float32
        )
        output.loss = torch.tensor(0.5)
        return output

    model = MagicMock(side_effect=fake_forward)
    model.eval = MagicMock(return_value=None)
    model.train = MagicMock(return_value=None)
    model.parameters = MagicMock(return_value=iter([torch.tensor([1.0])]))
    return model


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: augment_titles
# ═══════════════════════════════════════════════════════════════════════════

class TestAugmentTitles:

    def test_sempre_cria_tipo_a(self, sample_df_train):
        """Deve sempre criar exemplos tipo A (só título)."""
        result = augment_titles(sample_df_train)
        title_only = result[result["aug_type"] == "title_only"]
        assert len(title_only) == len(sample_df_train)

    def test_tipo_b_para_textos_medios(self, sample_df_train):
        """Tipo B (título + 50w) apenas para textos >= 10 words."""
        result = augment_titles(sample_df_train)
        title_50w = result[result["aug_type"] == "title_50w"]
        # artigos com texto >= 10 words: Fake_1 (150w), Real_1 (70w), Fake_2 (20w)
        assert len(title_50w) == 3

    def test_tipo_c_para_textos_longos(self, sample_df_train):
        """Tipo C (título + 100w) apenas para textos >= 50 words."""
        result = augment_titles(sample_df_train)
        title_100w = result[result["aug_type"] == "title_100w"]
        # artigos com texto >= 50 words: Fake_1 (150w), Real_1 (70w)
        assert len(title_100w) == 2

    def test_labels_preservados(self, sample_df_train):
        """Labels devem ser preservados em todos os exemplos aumentados."""
        result = augment_titles(sample_df_train)
        # verificar que cada artigo original tem os labels corretos
        for _, row in sample_df_train.iterrows():
            aug_rows = result[result["input_text"].str.startswith(
                str(row["title_clean"])
            )]
            assert all(aug_rows["label"] == row["label"])

    def test_aumenta_dataset(self, sample_df_train):
        """Dataset aumentado deve ter mais exemplos que o original."""
        result = augment_titles(sample_df_train)
        assert len(result) > len(sample_df_train)

    def test_colunas_corretas(self, sample_df_train):
        """Resultado deve ter input_text, label, aug_type."""
        result = augment_titles(sample_df_train)
        assert "input_text" in result.columns
        assert "label"      in result.columns
        assert "aug_type"   in result.columns

    def test_tipo_a_contem_so_titulo(self, sample_df_train):
        """Tipo A deve conter apenas o título sem [SEP]."""
        result = augment_titles(sample_df_train)
        title_only = result[result["aug_type"] == "title_only"]
        for _, row in title_only.iterrows():
            assert "[SEP]" not in row["input_text"]

    def test_tipos_b_c_contem_sep(self, sample_df_train):
        """Tipos B e C devem conter [SEP] entre título e texto."""
        result = augment_titles(sample_df_train)
        for aug_type in ["title_50w", "title_100w"]:
            rows = result[result["aug_type"] == aug_type]
            for _, row in rows.iterrows():
                assert "[SEP]" in row["input_text"]

    def test_texto_muito_curto_so_tipo_a(self):
        """Artigo com texto muito curto deve gerar só tipo A."""
        df = pd.DataFrame({
            "title_clean": ["Short title"],
            "text_clean":  ["tiny"],  # < 10 words
            "label":       [1],
        })
        result = augment_titles(df)
        assert len(result) == 1
        assert result.iloc[0]["aug_type"] == "title_only"

    def test_nan_texto_so_tipo_a(self):
        """Artigo com texto NaN deve gerar só tipo A."""
        df = pd.DataFrame({
            "title_clean": ["Title without text"],
            "text_clean":  [None],
            "label":       [0],
        })
        result = augment_titles(df)
        assert len(result) == 1
        assert result.iloc[0]["aug_type"] == "title_only"


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: TitlesDataset
# ═══════════════════════════════════════════════════════════════════════════

class TestTitlesDataset:

    def test_tamanho_correto(self, mock_tokenizer):
        """__len__ deve devolver o número de exemplos."""
        texts  = ["Title A", "Title B", "Title C"]
        labels = [1, 0, 1]
        dataset = TitlesDataset(texts, labels, mock_tokenizer)
        assert len(dataset) == 3

    def test_item_tem_campos_corretos(self, mock_tokenizer):
        """__getitem__ deve devolver input_ids, attention_mask, label."""
        texts  = ["Title A"]
        labels = [1]

        # mock para item único
        def single_tokenize(text, **kwargs):
            return {
                "input_ids":      torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
        tokenizer = MagicMock(side_effect=single_tokenize)

        dataset = TitlesDataset(texts, labels, tokenizer)
        item = dataset[0]

        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "label"          in item

    def test_label_e_tensor(self, mock_tokenizer):
        """Label deve ser um tensor."""
        def single_tokenize(text, **kwargs):
            return {
                "input_ids":      torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
        tokenizer = MagicMock(side_effect=single_tokenize)
        dataset = TitlesDataset(["Title"], [1], tokenizer)
        item = dataset[0]
        assert isinstance(item["label"], torch.Tensor)

    def test_label_valor_correto(self, mock_tokenizer):
        """Label deve ter o valor correto."""
        def single_tokenize(text, **kwargs):
            return {
                "input_ids":      torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
        tokenizer = MagicMock(side_effect=single_tokenize)
        dataset = TitlesDataset(["Title fake"], [1], tokenizer)
        item = dataset[0]
        assert item["label"].item() == 1


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: evaluate_on_test
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateOnTest:

    def test_devolve_metricas_obrigatorias(self, mock_model, mock_tokenizer, sample_df_test):
        """Deve devolver accuracy, precision, recall, f1, auc."""
        metrics = evaluate_on_test(
            mock_model, mock_tokenizer,
            sample_df_test, torch.device("cpu")
        )
        assert "accuracy"  in metrics
        assert "precision" in metrics
        assert "recall"    in metrics
        assert "f1"        in metrics
        assert "auc"       in metrics

    def test_metricas_entre_0_e_1(self, mock_model, mock_tokenizer, sample_df_test):
        """Métricas devem estar em [0,1]."""
        metrics = evaluate_on_test(
            mock_model, mock_tokenizer,
            sample_df_test, torch.device("cpu")
        )
        for k, v in metrics.items():
            if v is not None:
                assert 0.0 <= v <= 1.0, f"{k}={v} fora de [0,1]"

    def test_metricas_arredondadas(self, mock_model, mock_tokenizer, sample_df_test):
        """Métricas devem estar arredondadas a 4 casas decimais."""
        metrics = evaluate_on_test(
            mock_model, mock_tokenizer,
            sample_df_test, torch.device("cpu")
        )
        for k, v in metrics.items():
            if v is not None:
                assert v == round(v, 4), f"{k} não está arredondado"