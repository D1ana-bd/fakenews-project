"""
test_zero_shot.py
=================
Testes unitários para src/FKNWS/models/zero_shot.py

Cobre:
    - Carregamento do test.csv
    - Classificação zero-shot (mock do pipeline)
    - Conversão de scores para formato correto
    - Avaliação de métricas
    - Guardar resultados JSON
    - Edge cases: títulos vazios, labels incorretos

Uso:
    pytest tests/test_zero_shot.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from FKNWS.models.zero_shot import (
    classify_titles,
    convert_to_score,
    evaluate_zeroshot,
    load_test_data,
    save_results,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_test_csv(tmp_path):
    """Cria um test.csv temporário."""
    df = pd.DataFrame({
        "id":          ["Fake_1", "Real_1", "Fake_2", "Real_2", "Fake_3"],
        "title_clean": [
            "Government hiding secret cure for cancer",
            "Federal Reserve raises interest rates",
            "SHOCKING conspiracy revealed by whistleblower",
            "Scientists publish new climate research",
            "You won't believe what they found",
        ],
        "text_clean":  ["text1", "text2", "text3", "text4", "text5"],
        "label":       [1, 0, 1, 0, 1],
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_pipeline():
    """Mock do pipeline zero-shot que devolve resultados fixos."""
    def fake_classify(texts, candidate_labels, **kwargs):
        # se receber lista, devolve lista de resultados
        if isinstance(texts, list):
            results = []
            for text in texts:
                # simula: textos com "conspiracy" ou "secret" → fake
                if any(w in text.lower() for w in ["conspiracy", "secret", "shocking", "won't believe"]):
                    results.append({
                        "labels": ["fake news", "real news"],
                        "scores": [0.85, 0.15],
                    })
                else:
                    results.append({
                        "labels": ["fake news", "real news"],
                        "scores": [0.20, 0.80],
                    })
            return results
        else:
            # texto único
            return {
                "labels": ["fake news", "real news"],
                "scores": [0.75, 0.25],
            }

    return MagicMock(side_effect=fake_classify)


@pytest.fixture
def sample_zeroshot_results():
    """Resultados simulados do zero-shot para 5 artigos."""
    return [
        {"labels": ["fake news", "real news"], "scores": [0.85, 0.15]},
        {"labels": ["fake news", "real news"], "scores": [0.20, 0.80]},
        {"labels": ["fake news", "real news"], "scores": [0.78, 0.22]},
        {"labels": ["fake news", "real news"], "scores": [0.15, 0.85]},
        {"labels": ["fake news", "real news"], "scores": [0.90, 0.10]},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: load_test_data
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadTestData:

    def test_carrega_colunas_necessarias(self, sample_test_csv):
        """Deve carregar id, title_clean e label."""
        df = load_test_data(sample_test_csv)
        assert "id" in df.columns
        assert "title_clean" in df.columns
        assert "label" in df.columns

    def test_numero_linhas_correto(self, sample_test_csv):
        """Deve carregar todas as linhas."""
        df = load_test_data(sample_test_csv)
        assert len(df) == 5

    def test_labels_sao_binarios(self, sample_test_csv):
        """Labels devem ser 0 ou 1."""
        df = load_test_data(sample_test_csv)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_remove_titulos_vazios(self, tmp_path):
        """Deve remover linhas com título vazio ou NaN."""
        df = pd.DataFrame({
            "id":          ["Fake_1", "Real_1", "Fake_2"],
            "title_clean": ["Valid title", "", None],
            "label":       [1, 0, 1],
        })
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        result = load_test_data(path)
        assert len(result) == 1
        assert result.iloc[0]["id"] == "Fake_1"

    def test_erro_se_ficheiro_nao_existe(self):
        """Deve lançar erro se o ficheiro não existir."""
        with pytest.raises(FileNotFoundError):
            load_test_data(Path("/nao/existe/test.csv"))


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: convert_to_score
# ═══════════════════════════════════════════════════════════════════════════

class TestConvertToScore:

    def test_score_fake_quando_fake_primeiro(self):
        """Se 'fake news' for o primeiro label, score_fake = scores[0]."""
        result = {
            "labels": ["fake news", "real news"],
            "scores": [0.75, 0.25],
        }
        score = convert_to_score(result)
        assert abs(score - 0.75) < 1e-6

    def test_score_fake_quando_real_primeiro(self):
        """Se 'real news' for o primeiro label, score_fake = scores[1]."""
        result = {
            "labels": ["real news", "fake news"],
            "scores": [0.30, 0.70],
        }
        score = convert_to_score(result)
        assert abs(score - 0.70) < 1e-6

    def test_score_entre_0_e_1(self):
        """Score deve estar em [0,1]."""
        result = {
            "labels": ["fake news", "real news"],
            "scores": [0.60, 0.40],
        }
        score = convert_to_score(result)
        assert 0.0 <= score <= 1.0

    def test_scores_somam_1(self):
        """scores[0] + scores[1] deve ser ≈ 1.0."""
        result = {
            "labels": ["fake news", "real news"],
            "scores": [0.55, 0.45],
        }
        score = convert_to_score(result)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: classify_titles
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyTitles:

    def test_devolve_lista_do_mesmo_tamanho(self, mock_pipeline):
        """Deve devolver lista com mesmo número de títulos."""
        titles = ["Title A", "Title B", "Title C"]
        results = classify_titles(titles, mock_pipeline)
        assert len(results) == 3

    def test_cada_resultado_tem_labels_e_scores(self, mock_pipeline):
        """Cada resultado deve ter 'labels' e 'scores'."""
        titles = ["Title A"]
        results = classify_titles(titles, mock_pipeline)
        assert "labels" in results[0]
        assert "scores" in results[0]

    def test_scores_sao_floats(self, mock_pipeline):
        """Scores devem ser floats."""
        titles = ["Title A", "Title B"]
        results = classify_titles(titles, mock_pipeline)
        for r in results:
            for s in r["scores"]:
                assert isinstance(s, float)

    def test_lista_vazia_nao_da_erro(self, mock_pipeline):
        """Lista vazia deve devolver lista vazia."""
        results = classify_titles([], mock_pipeline)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: evaluate_zeroshot
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateZeroshot:

    def test_devolve_metricas_obrigatorias(self, sample_zeroshot_results):
        """Deve devolver accuracy, precision, recall, f1, auc."""
        y_true = [1, 0, 1, 0, 1]
        metrics = evaluate_zeroshot(y_true, sample_zeroshot_results)
        assert "accuracy"  in metrics
        assert "precision" in metrics
        assert "recall"    in metrics
        assert "f1"        in metrics
        assert "auc"       in metrics

    def test_metricas_entre_0_e_1(self, sample_zeroshot_results):
        """Todas as métricas devem estar em [0,1]."""
        y_true = [1, 0, 1, 0, 1]
        metrics = evaluate_zeroshot(y_true, sample_zeroshot_results)
        for k, v in metrics.items():
            if v is not None:
                assert 0.0 <= v <= 1.0, f"{k}={v} fora de [0,1]"

    def test_perfeito_da_f1_1(self):
        """Predição perfeita deve dar F1=1.0."""
        y_true = [1, 0, 1, 0]
        results = [
            {"labels": ["fake news", "real news"], "scores": [0.9, 0.1]},
            {"labels": ["fake news", "real news"], "scores": [0.1, 0.9]},
            {"labels": ["fake news", "real news"], "scores": [0.9, 0.1]},
            {"labels": ["fake news", "real news"], "scores": [0.1, 0.9]},
        ]
        metrics = evaluate_zeroshot(y_true, results)
        assert metrics["f1"] == 1.0

    def test_tamanho_y_true_igual_resultados(self, sample_zeroshot_results):
        """y_true e results devem ter o mesmo tamanho."""
        y_true = [1, 0, 1, 0, 1]
        assert len(y_true) == len(sample_zeroshot_results)


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: save_results
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveResults:

    def test_cria_ficheiro_json(self, tmp_path, sample_zeroshot_results):
        """Deve criar ficheiro JSON."""
        y_true = [1, 0, 1, 0, 1]
        metrics = {"accuracy": 0.8, "f1": 0.75, "auc": 0.82,
                   "precision": 0.77, "recall": 0.73}
        titles = ["T1", "T2", "T3", "T4", "T5"]
        path = tmp_path / "results_zeroshot.json"
        save_results(metrics, y_true, sample_zeroshot_results, titles, path)
        assert path.exists()

    def test_estrutura_json_correta(self, tmp_path, sample_zeroshot_results):
        """JSON deve ter campos obrigatórios."""
        y_true = [1, 0, 1, 0, 1]
        metrics = {"accuracy": 0.8, "f1": 0.75, "auc": 0.82,
                   "precision": 0.77, "recall": 0.73}
        titles = ["T1", "T2", "T3", "T4", "T5"]
        path = tmp_path / "results_zeroshot.json"
        save_results(metrics, y_true, sample_zeroshot_results, titles, path)

        with open(path) as f:
            data = json.load(f)

        assert "model"       in data
        assert "metrics"     in data
        assert "predictions" in data
        assert "n_articles"  in data

    def test_metricas_guardadas_corretamente(self, tmp_path, sample_zeroshot_results):
        """Métricas devem ser guardadas com valores corretos."""
        y_true = [1, 0, 1, 0, 1]
        metrics = {"accuracy": 0.8, "f1": 0.75, "auc": 0.82,
                   "precision": 0.77, "recall": 0.73}
        titles = ["T1", "T2", "T3", "T4", "T5"]
        path = tmp_path / "results_zeroshot.json"
        save_results(metrics, y_true, sample_zeroshot_results, titles, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metrics"]["f1"] == 0.75
        assert data["metrics"]["auc"] == 0.82

    def test_predictions_tem_campos_obrigatorios(self, tmp_path, sample_zeroshot_results):
        """Cada predição deve ter title, label, pred, score_fake."""
        y_true = [1, 0, 1, 0, 1]
        metrics = {"accuracy": 0.8, "f1": 0.75, "auc": 0.82,
                   "precision": 0.77, "recall": 0.73}
        titles = ["T1", "T2", "T3", "T4", "T5"]
        path = tmp_path / "results_zeroshot.json"
        save_results(metrics, y_true, sample_zeroshot_results, titles, path)

        with open(path) as f:
            data = json.load(f)

        for pred in data["predictions"]:
            assert "title"      in pred
            assert "label"      in pred
            assert "pred"       in pred
            assert "score_fake" in pred

    def test_cria_diretorias_se_nao_existem(self, tmp_path, sample_zeroshot_results):
        """Deve criar diretorias intermédias se não existirem."""
        y_true = [1, 0, 1, 0, 1]
        metrics = {"accuracy": 0.8, "f1": 0.75, "auc": 0.82,
                   "precision": 0.77, "recall": 0.73}
        titles = ["T1", "T2", "T3", "T4", "T5"]
        path = tmp_path / "subdir" / "results_zeroshot.json"
        save_results(metrics, y_true, sample_zeroshot_results, titles, path)
        assert path.exists()