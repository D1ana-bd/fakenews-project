"""
test_generate_predictions.py
============================
Testes unitários para src/FKNWS/integration/generate_predictions.py

Cobre:
    - Carregamento e filtragem dos CSVs gossipcop/politifact
    - Normalização e limpeza de títulos
    - Estrutura do output JSON
    - Lógica de inferência (mock do modelo)
    - Filtragem por artigos com métricas de rede
    - Edge cases: títulos vazios, duplicados, CSVs em falta

Uso:
    pytest tests/test_generate_predictions.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

# garantir que src/ está no path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from FKNWS.integration.generate_predictions import (
    load_gossipcop_politifact,
    predict_batch,
    run_inference,
    save_predictions,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """DataFrame simples com artigos fake e real."""
    return pd.DataFrame({
        "article_id": [
            "gossipcop-111",
            "gossipcop-222",
            "gossipcop-333",
            "politifact-444",
        ],
        "title": [
            "Celebrity spotted doing something shocking",
            "Local election results confirmed by officials",
            "Scientists discover miracle cure for everything",
            "Government releases new budget plan",
        ],
        "label": [1, 0, 1, 0],
    })


@pytest.fixture
def sample_df_with_preds(sample_df):
    """DataFrame com predições já calculadas."""
    df = sample_df.copy()
    df["pred_nlp"]  = [1, 0, 1, 0]
    df["score_nlp"] = [0.82, 0.23, 0.91, 0.18]
    return df


@pytest.fixture
def mock_model():
    """Mock do modelo BERT — side_effect devolve logits como tensores reais."""
    def fake_forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        logits_data = [[0.2, 2.5] if i % 2 == 0 else [2.5, 0.2] for i in range(batch_size)]
        output = MagicMock()
        output.logits = torch.tensor(logits_data, dtype=torch.float32)
        return output

    model = MagicMock(side_effect=fake_forward)
    model.eval.return_value = model
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock do tokenizer BERT — side_effect devolve tensores reais."""
    def fake_tokenize(texts, **kwargs):
        batch_size = len(texts)
        return {
            "input_ids":      torch.ones(batch_size, 10, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
        }
    return MagicMock(side_effect=fake_tokenize)


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: load_gossipcop_politifact
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadGossipcopPolitifact:

    def test_load_combina_fake_e_real(self, tmp_path):
        """Deve combinar artigos fake e real dos 4 CSVs."""
        # criar CSVs temporários
        gossipcop_fake = pd.DataFrame({
            "id":    ["gossipcop-1", "gossipcop-2"],
            "title": ["Fake title 1", "Fake title 2"],
        })
        gossipcop_real = pd.DataFrame({
            "id":    ["gossipcop-3"],
            "title": ["Real title 1"],
        })
        network_metrics = pd.DataFrame({
            "article_id": ["gossipcop-1", "gossipcop-2", "gossipcop-3"],
            "label":      ["fake", "fake", "real"],
        })

        gossipcop_fake.to_csv(tmp_path / "gossipcop_fake.csv", index=False)
        gossipcop_real.to_csv(tmp_path / "gossipcop_real.csv", index=False)
        network_metrics.to_csv(tmp_path / "network_metrics.csv", index=False)

        # ficheiros em falta (politifact) — não deve dar erro
        (tmp_path / "politifact_fake.csv").write_text("id,title\n")
        (tmp_path / "politifact_real.csv").write_text("id,title\n")

        with patch("FKNWS.integration.generate_predictions.RAW_DATA", tmp_path), \
             patch("FKNWS.integration.generate_predictions.NETWORK_METRICS_PATH",
                   tmp_path / "network_metrics.csv"):
            df = load_gossipcop_politifact()

        assert len(df) == 3
        assert set(df["article_id"]) == {"gossipcop-1", "gossipcop-2", "gossipcop-3"}

    def test_label_fake_e_1_real_e_0(self, tmp_path):
        """Fake deve ter label=1, Real label=0."""
        pd.DataFrame({"id": ["gossipcop-1"], "title": ["Fake news!"]}).to_csv(
            tmp_path / "gossipcop_fake.csv", index=False
        )
        pd.DataFrame({"id": ["gossipcop-2"], "title": ["Real news"]}).to_csv(
            tmp_path / "gossipcop_real.csv", index=False
        )
        pd.DataFrame({"article_id": ["gossipcop-1", "gossipcop-2"]}).to_csv(
            tmp_path / "network_metrics.csv", index=False
        )
        (tmp_path / "politifact_fake.csv").write_text("id,title\n")
        (tmp_path / "politifact_real.csv").write_text("id,title\n")

        with patch("FKNWS.integration.generate_predictions.RAW_DATA", tmp_path), \
             patch("FKNWS.integration.generate_predictions.NETWORK_METRICS_PATH",
                   tmp_path / "network_metrics.csv"):
            df = load_gossipcop_politifact()

        fake_row = df[df["article_id"] == "gossipcop-1"].iloc[0]
        real_row = df[df["article_id"] == "gossipcop-2"].iloc[0]
        assert fake_row["label"] == 1
        assert real_row["label"] == 0

    def test_remove_titulos_vazios(self, tmp_path):
        """Artigos com título vazio ou NaN devem ser removidos."""
        pd.DataFrame({
            "id":    ["gossipcop-1", "gossipcop-2", "gossipcop-3"],
            "title": ["Valid title", "", None],
        }).to_csv(tmp_path / "gossipcop_fake.csv", index=False)
        pd.DataFrame({"id": [], "title": []}).to_csv(
            tmp_path / "gossipcop_real.csv", index=False
        )
        pd.DataFrame({
            "article_id": ["gossipcop-1", "gossipcop-2", "gossipcop-3"]
        }).to_csv(tmp_path / "network_metrics.csv", index=False)
        (tmp_path / "politifact_fake.csv").write_text("id,title\n")
        (tmp_path / "politifact_real.csv").write_text("id,title\n")

        with patch("FKNWS.integration.generate_predictions.RAW_DATA", tmp_path), \
             patch("FKNWS.integration.generate_predictions.NETWORK_METRICS_PATH",
                   tmp_path / "network_metrics.csv"):
            df = load_gossipcop_politifact()

        assert len(df) == 1
        assert df.iloc[0]["article_id"] == "gossipcop-1"

    def test_remove_duplicados(self, tmp_path):
        """Artigos duplicados devem ser removidos (mesmo article_id)."""
        pd.DataFrame({
            "id":    ["gossipcop-1", "gossipcop-1"],
            "title": ["Title A", "Title B"],
        }).to_csv(tmp_path / "gossipcop_fake.csv", index=False)
        pd.DataFrame({"id": [], "title": []}).to_csv(
            tmp_path / "gossipcop_real.csv", index=False
        )
        pd.DataFrame({"article_id": ["gossipcop-1"]}).to_csv(
            tmp_path / "network_metrics.csv", index=False
        )
        (tmp_path / "politifact_fake.csv").write_text("id,title\n")
        (tmp_path / "politifact_real.csv").write_text("id,title\n")

        with patch("FKNWS.integration.generate_predictions.RAW_DATA", tmp_path), \
             patch("FKNWS.integration.generate_predictions.NETWORK_METRICS_PATH",
                   tmp_path / "network_metrics.csv"):
            df = load_gossipcop_politifact()

        assert len(df) == 1

    def test_filtra_artigos_sem_rede(self, tmp_path):
        """Artigos sem entrada no network_metrics.csv devem ser excluídos."""
        pd.DataFrame({
            "id":    ["gossipcop-1", "gossipcop-2", "gossipcop-3"],
            "title": ["Title 1", "Title 2", "Title 3"],
        }).to_csv(tmp_path / "gossipcop_fake.csv", index=False)
        pd.DataFrame({"id": [], "title": []}).to_csv(
            tmp_path / "gossipcop_real.csv", index=False
        )
        # só gossipcop-1 tem métricas de rede
        pd.DataFrame({"article_id": ["gossipcop-1"]}).to_csv(
            tmp_path / "network_metrics.csv", index=False
        )
        (tmp_path / "politifact_fake.csv").write_text("id,title\n")
        (tmp_path / "politifact_real.csv").write_text("id,title\n")

        with patch("FKNWS.integration.generate_predictions.RAW_DATA", tmp_path), \
             patch("FKNWS.integration.generate_predictions.NETWORK_METRICS_PATH",
                   tmp_path / "network_metrics.csv"):
            df = load_gossipcop_politifact()

        assert len(df) == 1
        assert df.iloc[0]["article_id"] == "gossipcop-1"


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: predict_batch
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictBatch:

    def test_output_tem_tamanho_correto(self, mock_model, mock_tokenizer):
        """Deve devolver listas do mesmo tamanho que o input."""
        titles = ["Title A", "Title B", "Title C"]
        preds, scores = predict_batch(
            titles, mock_model, mock_tokenizer, torch.device("cpu")
        )
        assert len(preds) == 3
        assert len(scores) == 3

    def test_scores_entre_0_e_1(self, mock_model, mock_tokenizer):
        """Scores de probabilidade devem estar em [0,1]."""
        titles = ["Title A", "Title B"]
        _, scores = predict_batch(
            titles, mock_model, mock_tokenizer, torch.device("cpu")
        )
        for score in scores:
            assert 0.0 <= score <= 1.0, f"Score fora de [0,1]: {score}"

    def test_preds_sao_binarios(self, mock_model, mock_tokenizer):
        """Predições devem ser 0 ou 1."""
        titles = ["Title A", "Title B", "Title C"]
        preds, _ = predict_batch(
            titles, mock_model, mock_tokenizer, torch.device("cpu")
        )
        for pred in preds:
            assert pred in (0, 1), f"Predição inválida: {pred}"

    def test_batch_vazio_nao_da_erro(self):
        """Batch vazio deve devolver listas vazias sem erros."""
        empty_output = MagicMock()
        empty_output.logits = torch.zeros(0, 2, dtype=torch.float32)
        model = MagicMock(return_value=empty_output)
        model.eval.return_value = model

        def empty_tokenize(texts, **kwargs):
            return {
                "input_ids":      torch.zeros(0, 10, dtype=torch.long),
                "attention_mask": torch.zeros(0, 10, dtype=torch.long),
            }
        tokenizer = MagicMock(side_effect=empty_tokenize)

        preds, scores = predict_batch([], model, tokenizer, torch.device("cpu"))
        assert preds == []
        assert scores == []


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: run_inference
# ═══════════════════════════════════════════════════════════════════════════

class TestRunInference:

    def test_adiciona_colunas_pred_e_score(self, sample_df, mock_model, mock_tokenizer):
        """Deve adicionar colunas pred_nlp e score_nlp ao DataFrame."""
        result = run_inference(
            sample_df, mock_model, mock_tokenizer,
            torch.device("cpu"), batch_size=2
        )
        assert "pred_nlp" in result.columns
        assert "score_nlp" in result.columns

    def test_nao_modifica_df_original(self, sample_df, mock_model, mock_tokenizer):
        """Não deve modificar o DataFrame original (deve fazer .copy())."""
        original_cols = list(sample_df.columns)
        run_inference(
            sample_df, mock_model, mock_tokenizer,
            torch.device("cpu"), batch_size=2
        )
        assert list(sample_df.columns) == original_cols

    def test_todos_artigos_classificados(self, sample_df, mock_model, mock_tokenizer):
        """Todos os artigos devem ter uma predição."""
        result = run_inference(
            sample_df, mock_model, mock_tokenizer,
            torch.device("cpu"), batch_size=2
        )
        assert result["pred_nlp"].notna().all()
        assert result["score_nlp"].notna().all()
        assert len(result) == len(sample_df)

    def test_batch_size_nao_afeta_resultados(self, sample_df):
        """Batch size diferente deve dar os mesmos resultados."""
        # mock determinístico: sempre devolve Fake, independente do índice
        def det_forward(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            output = MagicMock()
            output.logits = torch.tensor([[0.2, 2.5]] * batch_size, dtype=torch.float32)
            return output

        def det_tokenize(texts, **kwargs):
            return {
                "input_ids":      torch.ones(len(texts), 10, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 10, dtype=torch.long),
            }

        model     = MagicMock(side_effect=det_forward)
        tokenizer = MagicMock(side_effect=det_tokenize)

        result_1 = run_inference(sample_df, model, tokenizer, torch.device("cpu"), batch_size=1)
        result_4 = run_inference(sample_df, model, tokenizer, torch.device("cpu"), batch_size=4)
        assert list(result_1["pred_nlp"]) == list(result_4["pred_nlp"])


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: save_predictions
# ═══════════════════════════════════════════════════════════════════════════

class TestSavePredictions:

    def test_cria_ficheiro_json(self, tmp_path, sample_df_with_preds):
        """Deve criar o ficheiro JSON no caminho especificado."""
        output_path = tmp_path / "predictions_bert.json"
        save_predictions(sample_df_with_preds, output_path)
        assert output_path.exists()

    def test_estrutura_json_correta(self, tmp_path, sample_df_with_preds):
        """JSON deve ter campos obrigatórios."""
        output_path = tmp_path / "predictions_bert.json"
        save_predictions(sample_df_with_preds, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "predictions" in data
        assert "n_articles" in data
        assert "model" in data
        assert data["n_articles"] == len(sample_df_with_preds)

    def test_cada_predicao_tem_campos_obrigatorios(self, tmp_path, sample_df_with_preds):
        """Cada predição deve ter article_id, label, pred_nlp, score_nlp."""
        output_path = tmp_path / "predictions_bert.json"
        save_predictions(sample_df_with_preds, output_path)

        with open(output_path) as f:
            data = json.load(f)

        for pred in data["predictions"]:
            assert "article_id" in pred
            assert "label" in pred
            assert "pred_nlp" in pred
            assert "score_nlp" in pred

    def test_scores_sao_floats(self, tmp_path, sample_df_with_preds):
        """Scores devem ser floats em [0,1]."""
        output_path = tmp_path / "predictions_bert.json"
        save_predictions(sample_df_with_preds, output_path)

        with open(output_path) as f:
            data = json.load(f)

        for pred in data["predictions"]:
            assert isinstance(pred["score_nlp"], float)
            assert 0.0 <= pred["score_nlp"] <= 1.0

    def test_labels_sao_inteiros(self, tmp_path, sample_df_with_preds):
        """Labels e predições devem ser inteiros 0 ou 1."""
        output_path = tmp_path / "predictions_bert.json"
        save_predictions(sample_df_with_preds, output_path)

        with open(output_path) as f:
            data = json.load(f)

        for pred in data["predictions"]:
            assert pred["label"] in (0, 1)
            assert pred["pred_nlp"] in (0, 1)

    def test_cria_diretorias_se_nao_existem(self, tmp_path, sample_df_with_preds):
        """Deve criar diretorias intermédias se não existirem."""
        output_path = tmp_path / "subdir" / "nested" / "predictions.json"
        save_predictions(sample_df_with_preds, output_path)
        assert output_path.exists()

    def test_n_articles_correto(self, tmp_path, sample_df_with_preds):
        """n_articles deve corresponder ao número de predições."""
        output_path = tmp_path / "predictions_bert.json"
        save_predictions(sample_df_with_preds, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["n_articles"] == len(data["predictions"])