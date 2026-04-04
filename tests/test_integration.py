"""
test_integration.py
===================
Testes unitários para src/FKNWS/integration/integration.py

Cobre:
    - Carregamento de predictions_bert.json
    - Carregamento de network_metrics.csv
    - Construção do score_network (normalização, inversão modularity)
    - Cálculo do ensemble (diferentes alphas)
    - Avaliação das 3 abordagens
    - Tabela comparativa
    - Pipeline completo (run_integration)
    - Edge cases: join vazio, features em falta, alpha extremos

Uso:
    pytest tests/test_integration.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.FKNWS.integration.integration import (
    build_comparison_table,
    build_network_score,
    compute_ensemble,
    evaluate,
    evaluate_all_approaches,
    load_network_metrics,
    load_nlp_results,
    run_integration,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_predictions_json(tmp_path):
    """Cria um predictions_bert.json temporário."""
    data = {
        "model": "bert_finetuned_cascade",
        "input": "title_only",
        "n_articles": 6,
        "predictions": [
            {"article_id": "gossipcop-1", "label": 1, "pred_nlp": 1, "score_nlp": 0.82},
            {"article_id": "gossipcop-2", "label": 0, "pred_nlp": 0, "score_nlp": 0.21},
            {"article_id": "gossipcop-3", "label": 1, "pred_nlp": 1, "score_nlp": 0.75},
            {"article_id": "gossipcop-4", "label": 0, "pred_nlp": 0, "score_nlp": 0.18},
            {"article_id": "gossipcop-5", "label": 1, "pred_nlp": 0, "score_nlp": 0.45},
            {"article_id": "gossipcop-6", "label": 0, "pred_nlp": 1, "score_nlp": 0.55},
        ],
    }
    path = tmp_path / "predictions_bert.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_network_csv(tmp_path):
    """Cria um network_metrics.csv temporário."""
    df = pd.DataFrame({
        "article_id":             ["gossipcop-1","gossipcop-2","gossipcop-3",
                                   "gossipcop-4","gossipcop-5","gossipcop-6"],
        "label":                  ["fake","real","fake","real","fake","real"],
        "degree_centrality_mean": [0.20, 0.05, 0.18, 0.04, 0.22, 0.06],
        "betweenness_mean":       [0.12, 0.03, 0.10, 0.02, 0.14, 0.03],
        "propagation_speed":      [15.0, 8.0,  14.0, 7.0,  16.0, 9.0],
        "modularity":             [0.43, 0.58, 0.45, 0.60, 0.41, 0.55],
        "n_communities":          [8.0,  7.0,  9.0,  6.0,  8.0,  7.0],
    })
    path = tmp_path / "network_metrics.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_df_joined():
    """DataFrame já com score_nlp e métricas de rede (após join)."""
    return pd.DataFrame({
        "article_id":             ["gossipcop-1","gossipcop-2","gossipcop-3",
                                   "gossipcop-4","gossipcop-5","gossipcop-6"],
        "label":                  [1, 0, 1, 0, 1, 0],
        "pred_nlp":               [1, 0, 1, 0, 0, 1],
        "score_nlp":              [0.82, 0.21, 0.75, 0.18, 0.45, 0.55],
        "degree_centrality_mean": [0.20, 0.05, 0.18, 0.04, 0.22, 0.06],
        "betweenness_mean":       [0.12, 0.03, 0.10, 0.02, 0.14, 0.03],
        "propagation_speed":      [15.0, 8.0,  14.0, 7.0,  16.0, 9.0],
        "modularity":             [0.43, 0.58, 0.45, 0.60, 0.41, 0.55],
        "n_communities":          [8.0,  7.0,  9.0,  6.0,  8.0,  7.0],
    })


@pytest.fixture
def sample_df_with_network_score(sample_df_joined):
    """DataFrame com score_network já calculado."""
    df = sample_df_joined.copy()
    df["score_network"] = build_network_score(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: load_nlp_results
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadNlpResults:

    def test_carrega_colunas_corretas(self, sample_predictions_json):
        """Deve devolver DataFrame com colunas article_id, label, score_nlp, pred_nlp."""
        df = load_nlp_results(sample_predictions_json)
        assert set(df.columns) == {"article_id", "label", "score_nlp", "pred_nlp"}

    def test_numero_artigos_correto(self, sample_predictions_json):
        """Deve carregar todos os artigos do JSON."""
        df = load_nlp_results(sample_predictions_json)
        assert len(df) == 6

    def test_scores_sao_floats(self, sample_predictions_json):
        """Scores NLP devem ser floats."""
        df = load_nlp_results(sample_predictions_json)
        assert df["score_nlp"].dtype == float

    def test_labels_sao_inteiros(self, sample_predictions_json):
        """Labels devem ser 0 ou 1."""
        df = load_nlp_results(sample_predictions_json)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_erro_se_predictions_vazio(self, tmp_path):
        """Deve lançar ValueError se predictions estiver vazio."""
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"predictions": []}))
        with pytest.raises(ValueError, match="vazio ou ausente"):
            load_nlp_results(path)

    def test_erro_se_colunas_em_falta(self, tmp_path):
        """Deve lançar ValueError se faltar coluna obrigatória."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"predictions": [{"article_id": "x", "label": 1}]}))
        with pytest.raises(ValueError, match="Colunas em falta"):
            load_nlp_results(path)


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: load_network_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadNetworkMetrics:

    def test_carrega_article_id(self, sample_network_csv):
        """Deve ter coluna article_id."""
        df = load_network_metrics(sample_network_csv)
        assert "article_id" in df.columns

    def test_carrega_features_disponiveis(self, sample_network_csv):
        """Deve carregar as features de rede disponíveis."""
        df = load_network_metrics(sample_network_csv)
        assert "degree_centrality_mean" in df.columns
        assert "modularity" in df.columns

    def test_numero_artigos_correto(self, sample_network_csv):
        """Deve carregar todos os artigos."""
        df = load_network_metrics(sample_network_csv)
        assert len(df) == 6

    def test_renomeia_id_para_article_id(self, tmp_path):
        """Se a coluna se chamar 'id', deve renomear para 'article_id'."""
        df = pd.DataFrame({
            "id": ["gossipcop-1"],
            "degree_centrality_mean": [0.1],
            "betweenness_mean": [0.05],
            "propagation_speed": [10.0],
            "modularity": [0.5],
            "n_communities": [7.0],
        })
        path = tmp_path / "net.csv"
        df.to_csv(path, index=False)
        result = load_network_metrics(path)
        assert "article_id" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: build_network_score
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildNetworkScore:

    def test_scores_entre_0_e_1(self, sample_df_joined):
        """Todos os scores devem estar em [0,1]."""
        scores = build_network_score(sample_df_joined)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_tamanho_igual_ao_input(self, sample_df_joined):
        """Score deve ter o mesmo número de elementos que o DataFrame."""
        scores = build_network_score(sample_df_joined)
        assert len(scores) == len(sample_df_joined)

    def test_fake_tem_score_mais_alto(self, sample_df_joined):
        """Artigos fake devem ter score_network médio mais alto que reais."""
        scores = build_network_score(sample_df_joined)
        fake_mean = scores[sample_df_joined["label"] == 1].mean()
        real_mean = scores[sample_df_joined["label"] == 0].mean()
        assert fake_mean > real_mean, (
            f"Fake score médio ({fake_mean:.3f}) deve ser > Real ({real_mean:.3f})"
        )

    def test_modularity_invertida(self):
        """Modularity alta deve resultar em score_network mais baixo."""
        # dois artigos: um com modularity alta (real), outro com baixa (fake)
        df = pd.DataFrame({
            "degree_centrality_mean": [0.2,  0.2],
            "betweenness_mean":       [0.1,  0.1],
            "propagation_speed":      [10.0, 10.0],
            "modularity":             [0.8,  0.2],   # alto vs baixo
            "n_communities":          [7.0,  7.0],
        })
        scores = build_network_score(df)
        # modularity alta (idx 0) → score mais baixo
        assert scores.iloc[0] < scores.iloc[1]

    def test_features_em_falta_nao_da_erro(self):
        """Deve funcionar mesmo sem todas as features (re-normaliza pesos)."""
        df = pd.DataFrame({
            "degree_centrality_mean": [0.2, 0.05],
            "betweenness_mean":       [0.1, 0.03],
            # sem propagation_speed, modularity, n_communities
        })
        scores = build_network_score(df)
        assert len(scores) == 2
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: compute_ensemble
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeEnsemble:

    def test_adiciona_colunas_score_final_e_pred(self, sample_df_with_network_score):
        """Deve adicionar score_final e pred_ensemble."""
        df = compute_ensemble(sample_df_with_network_score, alpha=0.6)
        assert "score_final" in df.columns
        assert "pred_ensemble" in df.columns

    def test_formula_alpha(self, sample_df_with_network_score):
        """score_final = α × score_nlp + (1-α) × score_network."""
        alpha = 0.7
        df = compute_ensemble(sample_df_with_network_score, alpha=alpha)
        expected = alpha * df["score_nlp"] + (1 - alpha) * df["score_network"]
        pd.testing.assert_series_equal(
            df["score_final"].round(6),
            expected.round(6),
            check_names=False,
        )

    def test_alpha_1_igual_ao_nlp(self, sample_df_with_network_score):
        """Com α=1, score_final deve ser igual ao score_nlp."""
        df = compute_ensemble(sample_df_with_network_score, alpha=1.0)
        pd.testing.assert_series_equal(
            df["score_final"].round(6),
            df["score_nlp"].round(6),
            check_names=False,
        )

    def test_alpha_0_igual_a_rede(self, sample_df_with_network_score):
        """Com α=0, score_final deve ser igual ao score_network."""
        df = compute_ensemble(sample_df_with_network_score, alpha=0.0)
        pd.testing.assert_series_equal(
            df["score_final"].round(6),
            df["score_network"].round(6),
            check_names=False,
        )

    def test_pred_ensemble_binario(self, sample_df_with_network_score):
        """Predições devem ser 0 ou 1."""
        df = compute_ensemble(sample_df_with_network_score, alpha=0.6)
        assert set(df["pred_ensemble"].unique()).issubset({0, 1})

    def test_nao_modifica_df_original(self, sample_df_with_network_score):
        """Não deve modificar o DataFrame original."""
        original_cols = list(sample_df_with_network_score.columns)
        compute_ensemble(sample_df_with_network_score, alpha=0.6)
        assert list(sample_df_with_network_score.columns) == original_cols

    def test_diferentes_alphas_dao_resultados_diferentes(self, sample_df_with_network_score):
        """Alphas diferentes devem produzir scores diferentes."""
        df_05 = compute_ensemble(sample_df_with_network_score, alpha=0.5)
        df_08 = compute_ensemble(sample_df_with_network_score, alpha=0.8)
        assert not df_05["score_final"].equals(df_08["score_final"])


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: evaluate
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_devolve_metricas_obrigatorias(self):
        """Deve devolver accuracy, precision, recall, f1."""
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 1, 0]
        metrics = evaluate(y_true, y_pred)
        assert "accuracy"  in metrics
        assert "precision" in metrics
        assert "recall"    in metrics
        assert "f1"        in metrics

    def test_metricas_entre_0_e_1(self):
        """Todas as métricas devem estar em [0,1]."""
        y_true = [1, 0, 1, 0, 1, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1, 1]
        metrics = evaluate(y_true, y_pred)
        for k, v in metrics.items():
            if k != "auc":
                assert 0.0 <= v <= 1.0, f"{k}={v} fora de [0,1]"

    def test_perfeito_da_f1_1(self):
        """Predição perfeita deve dar F1=1.0."""
        y_true = [1, 0, 1, 0]
        metrics = evaluate(y_true, y_true)
        assert metrics["f1"] == 1.0

    def test_auc_calculado_com_scores(self):
        """AUC deve ser calculado quando y_score é fornecido."""
        y_true  = [1, 0, 1, 0]
        y_pred  = [1, 0, 1, 0]
        y_score = [0.9, 0.1, 0.8, 0.2]
        metrics = evaluate(y_true, y_pred, y_score)
        assert "auc" in metrics
        assert metrics["auc"] is not None

    def test_valores_arredondados_a_4_casas(self):
        """Métricas devem ser arredondadas a 4 casas decimais."""
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        metrics = evaluate(y_true, y_pred)
        for k, v in metrics.items():
            if v is not None:
                assert v == round(v, 4), f"{k} não está arredondado a 4 casas"


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: evaluate_all_approaches
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateAllApproaches:

    def test_devolve_3_abordagens(self, sample_df_with_network_score):
        """Deve devolver resultados para network_only, nlp_only, ensemble."""
        df = compute_ensemble(sample_df_with_network_score, alpha=0.6)
        results = evaluate_all_approaches(df, alpha=0.6)
        assert "network_only" in results
        assert "nlp_only"     in results
        assert "ensemble"     in results

    def test_cada_abordagem_tem_f1(self, sample_df_with_network_score):
        """Cada abordagem deve ter métrica f1."""
        df = compute_ensemble(sample_df_with_network_score, alpha=0.6)
        results = evaluate_all_approaches(df, alpha=0.6)
        for approach in ["network_only", "nlp_only", "ensemble"]:
            assert "f1" in results[approach]


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: build_comparison_table
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildComparisonTable:

    def _make_all_results(self):
        metrics = {"accuracy": 0.7, "precision": 0.65, "recall": 0.70, "f1": 0.67}
        return {
            0.5: {"network_only": metrics, "nlp_only": metrics, "ensemble": metrics},
            0.6: {"network_only": metrics, "nlp_only": metrics,
                  "ensemble": {**metrics, "f1": 0.70}},
        }

    def test_tem_linha_por_abordagem(self):
        """Deve ter linha para Só Rede, Só NLP e cada alpha do Ensemble."""
        table = build_comparison_table(self._make_all_results())
        abordagens = table["abordagem"].tolist()
        assert "Só Rede"      in abordagens
        assert "Só NLP (BERT)" in abordagens
        assert abordagens.count("Ensemble") == 2   # um por alpha

    def test_tem_coluna_f1(self):
        """Tabela deve ter coluna f1."""
        table = build_comparison_table(self._make_all_results())
        assert "f1" in table.columns

    def test_tem_coluna_alpha(self):
        """Tabela deve ter coluna alpha."""
        table = build_comparison_table(self._make_all_results())
        assert "alpha" in table.columns


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: run_integration (pipeline completo)
# ═══════════════════════════════════════════════════════════════════════════

class TestRunIntegration:

    def test_pipeline_completo(self, sample_predictions_json, sample_network_csv, tmp_path):
        """Pipeline completo deve correr sem erros e guardar resultados."""
        with patch("FKNWS.integration.integration.NLP_RESULTS_PATH", sample_predictions_json), \
             patch("FKNWS.integration.integration.NETWORK_METRICS_PATH", sample_network_csv):
            results = run_integration(alphas=[0.5, 0.6], results_dir=tmp_path)

        assert "best_alpha"       in results
        assert "best_f1_ensemble" in results
        assert "nlp_only_f1"      in results
        assert "all_results"      in results

    def test_guarda_json(self, sample_predictions_json, sample_network_csv, tmp_path):
        """Deve guardar results_integration.json."""
        with patch("FKNWS.integration.integration.NLP_RESULTS_PATH", sample_predictions_json), \
             patch("FKNWS.integration.integration.NETWORK_METRICS_PATH", sample_network_csv):
            run_integration(alphas=[0.5], results_dir=tmp_path)

        assert (tmp_path / "results_integration.json").exists()

    def test_guarda_csv_comparativo(self, sample_predictions_json, sample_network_csv, tmp_path):
        """Deve guardar integration_comparison.csv."""
        with patch("FKNWS.integration.integration.NLP_RESULTS_PATH", sample_predictions_json), \
             patch("FKNWS.integration.integration.NETWORK_METRICS_PATH", sample_network_csv):
            run_integration(alphas=[0.5], results_dir=tmp_path)

        assert (tmp_path / "integration_comparison.csv").exists()

    def test_melhor_alpha_esta_nos_testados(self, sample_predictions_json, sample_network_csv, tmp_path):
        """best_alpha deve ser um dos alphas testados."""
        alphas = [0.5, 0.6, 0.7]
        with patch("FKNWS.integration.integration.NLP_RESULTS_PATH", sample_predictions_json), \
             patch("FKNWS.integration.integration.NETWORK_METRICS_PATH", sample_network_csv):
            results = run_integration(alphas=alphas, results_dir=tmp_path)

        assert results["best_alpha"] in alphas

    def test_delta_f1_calculado(self, sample_predictions_json, sample_network_csv, tmp_path):
        """delta_f1 deve ser best_f1_ensemble - nlp_only_f1."""
        with patch("FKNWS.integration.integration.NLP_RESULTS_PATH", sample_predictions_json), \
             patch("FKNWS.integration.integration.NETWORK_METRICS_PATH", sample_network_csv):
            results = run_integration(alphas=[0.5], results_dir=tmp_path)

        expected_delta = round(results["best_f1_ensemble"] - results["nlp_only_f1"], 4)
        assert results["delta_f1"] == expected_delta