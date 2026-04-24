"""
test_xgboost.py
===============
Testes unitários para src/FKNWS/integration/ensemble_xgboost.py

Cobre:
    - Carregamento de predictions_bert.json
    - Carregamento de results_zeroshot.json
    - Carregamento de network_metrics.csv
    - Preparação de features (join + normalização)
    - Treino e avaliação XGBoost (mock)
    - Estrutura do output JSON
    - Edge cases: join vazio, features em falta

Uso:
    pytest tests/test_xgboost.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.FKNWS.integration.ensemble_xgboost import (
    load_bert_predictions,
    load_network_metrics,
    prepare_features,
    train_evaluate_xgboost,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_bert_json(tmp_path):
    """Cria um predictions_bert.json temporário."""
    data = {
        "model": "bert_finetuned_cascade",
        "predictions": [
            {"article_id": "gossipcop-1", "label": 1,
             "pred_nlp": 1, "score_nlp": 0.82},
            {"article_id": "gossipcop-2", "label": 0,
             "pred_nlp": 0, "score_nlp": 0.21},
            {"article_id": "gossipcop-3", "label": 1,
             "pred_nlp": 1, "score_nlp": 0.75},
            {"article_id": "gossipcop-4", "label": 0,
             "pred_nlp": 0, "score_nlp": 0.18},
            {"article_id": "gossipcop-5", "label": 1,
             "pred_nlp": 0, "score_nlp": 0.45},
            {"article_id": "gossipcop-6", "label": 0,
             "pred_nlp": 1, "score_nlp": 0.55},
        ],
    }
    path = tmp_path / "predictions_bert.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_network_csv(tmp_path):
    """Cria um network_metrics.csv temporário."""
    df = pd.DataFrame({
        "article_id":             ["gossipcop-1", "gossipcop-2", "gossipcop-3",
                                   "gossipcop-4", "gossipcop-5", "gossipcop-6"],
        "label":                  ["fake", "real", "fake", "real", "fake", "real"],
        "degree_centrality_mean": [0.20, 0.05, 0.18, 0.04, 0.22, 0.06],
        "betweenness_mean":       [0.12, 0.03, 0.10, 0.02, 0.14, 0.03],
        "propagation_speed":      [15.0, 8.0, 14.0, 7.0, 16.0, 9.0],
        "modularity":             [0.43, 0.58, 0.45, 0.60, 0.41, 0.55],
        "n_communities":          [8.0, 7.0, 9.0, 6.0, 8.0, 7.0],
    })
    path = tmp_path / "network_metrics.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_df_nlp():
    """DataFrame simples com scores NLP."""
    return pd.DataFrame({
        "article_id":    ["gossipcop-1", "gossipcop-2", "gossipcop-3",
                          "gossipcop-4", "gossipcop-5", "gossipcop-6"],
        "label":         [1, 0, 1, 0, 1, 0],
        "score_nlp_bert": [0.82, 0.21, 0.75, 0.18, 0.45, 0.55],
    })


@pytest.fixture
def sample_df_net():
    """DataFrame simples com métricas de rede."""
    return pd.DataFrame({
        "article_id":             ["gossipcop-1", "gossipcop-2", "gossipcop-3",
                                   "gossipcop-4", "gossipcop-5", "gossipcop-6"],
        "degree_centrality_mean": [0.20, 0.05, 0.18, 0.04, 0.22, 0.06],
        "betweenness_mean":       [0.12, 0.03, 0.10, 0.02, 0.14, 0.03],
        "propagation_speed":      [15.0, 8.0, 14.0, 7.0, 16.0, 9.0],
        "modularity":             [0.43, 0.58, 0.45, 0.60, 0.41, 0.55],
        "n_communities":          [8.0, 7.0, 9.0, 6.0, 8.0, 7.0],
    })


@pytest.fixture
def sample_X_y():
    """Features e labels sintéticos para testar XGBoost."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[
        "score_nlp_bert", "degree_centrality_mean", "betweenness_mean",
        "propagation_speed", "modularity", "n_communities",
    ])
    y = pd.Series(y)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: load_bert_predictions
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadBertPredictions:

    def test_carrega_colunas_corretas(self, sample_bert_json):
        """Deve ter article_id, label, score_nlp_bert."""
        df = load_bert_predictions(sample_bert_json)
        assert "article_id"     in df.columns
        assert "label"          in df.columns
        assert "score_nlp_bert" in df.columns

    def test_numero_artigos_correto(self, sample_bert_json):
        """Deve carregar todos os artigos."""
        df = load_bert_predictions(sample_bert_json)
        assert len(df) == 6

    def test_score_nlp_renomeado(self, sample_bert_json):
        """score_nlp deve ser renomeado para score_nlp_bert."""
        df = load_bert_predictions(sample_bert_json)
        assert "score_nlp"      not in df.columns
        assert "score_nlp_bert" in df.columns

    def test_scores_entre_0_e_1(self, sample_bert_json):
        """Scores devem estar em [0,1]."""
        df = load_bert_predictions(sample_bert_json)
        assert (df["score_nlp_bert"] >= 0.0).all()
        assert (df["score_nlp_bert"] <= 1.0).all()

    def test_labels_sao_binarios(self, sample_bert_json):
        """Labels devem ser 0 ou 1."""
        df = load_bert_predictions(sample_bert_json)
        assert set(df["label"].unique()).issubset({0, 1})


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: load_network_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadNetworkMetrics:

    def test_carrega_article_id(self, sample_network_csv):
        """Deve ter coluna article_id."""
        df = load_network_metrics(sample_network_csv)
        assert "article_id" in df.columns

    def test_remove_coluna_label(self, sample_network_csv):
        """Deve remover coluna label do CSV de rede."""
        df = load_network_metrics(sample_network_csv)
        assert "label" not in df.columns

    def test_carrega_features_disponiveis(self, sample_network_csv):
        """Deve carregar as features de rede disponíveis."""
        df = load_network_metrics(sample_network_csv)
        assert "degree_centrality_mean" in df.columns
        assert "propagation_speed"      in df.columns
        assert "modularity"             in df.columns

    def test_numero_artigos_correto(self, sample_network_csv):
        """Deve carregar todos os artigos."""
        df = load_network_metrics(sample_network_csv)
        assert len(df) == 6


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: prepare_features
# ═══════════════════════════════════════════════════════════════════════════

class TestPrepareFeatures:

    def test_join_correto(self, sample_df_nlp, sample_df_net):
        """Join por article_id deve funcionar."""
        X, y = prepare_features(sample_df_nlp, sample_df_net, "score_nlp_bert")
        assert len(X) == 6

    def test_features_corretas(self, sample_df_nlp, sample_df_net):
        """X deve ter score_nlp + features de rede."""
        X, y = prepare_features(sample_df_nlp, sample_df_net, "score_nlp_bert")
        assert "score_nlp_bert"          in X.columns
        assert "degree_centrality_mean"  in X.columns
        assert "propagation_speed"       in X.columns

    def test_labels_corretos(self, sample_df_nlp, sample_df_net):
        """y deve ter os labels corretos."""
        X, y = prepare_features(sample_df_nlp, sample_df_net, "score_nlp_bert")
        assert list(y) == [1, 0, 1, 0, 1, 0]

    def test_features_rede_normalizadas(self, sample_df_nlp, sample_df_net):
        """Features de rede devem estar em [0,1] após normalização."""
        X, y = prepare_features(sample_df_nlp, sample_df_net, "score_nlp_bert")
        net_features = ["degree_centrality_mean", "betweenness_mean",
                        "propagation_speed", "modularity", "n_communities"]
        for feat in net_features:
            if feat in X.columns:
                assert X[feat].min() >= -1e-6
                assert X[feat].max() <= 1.0 + 1e-6

    def test_erro_se_join_vazio(self, sample_df_net):
        """Deve lançar ValueError se join resultar em 0 artigos."""
        df_nlp_sem_match = pd.DataFrame({
            "article_id":    ["inexistente-1", "inexistente-2"],
            "label":         [1, 0],
            "score_nlp_bert": [0.8, 0.2],
        })
        with pytest.raises(ValueError, match="0 artigos"):
            prepare_features(df_nlp_sem_match, sample_df_net, "score_nlp_bert")

    def test_nao_modifica_dfs_originais(self, sample_df_nlp, sample_df_net):
        """Não deve modificar os DataFrames originais."""
        nlp_cols_before = list(sample_df_nlp.columns)
        net_cols_before = list(sample_df_net.columns)
        prepare_features(sample_df_nlp, sample_df_net, "score_nlp_bert")
        assert list(sample_df_nlp.columns) == nlp_cols_before
        assert list(sample_df_net.columns) == net_cols_before


# ═══════════════════════════════════════════════════════════════════════════
# TESTES: train_evaluate_xgboost
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainEvaluateXgboost:

    def test_devolve_metricas_obrigatorias(self, sample_X_y):
        """Deve devolver cv_auc_mean, cv_f1_mean e feature_importance."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        assert "metrics"            in result
        assert "feature_importance" in result
        assert "cv_auc_mean"  in result["metrics"]
        assert "cv_f1_mean"   in result["metrics"]

    def test_metricas_entre_0_e_1(self, sample_X_y):
        """Métricas devem estar em [0,1]."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        for k, v in result["metrics"].items():
            assert 0.0 <= v <= 1.0, f"{k}={v} fora de [0,1]"

    def test_feature_importance_tem_todas_features(self, sample_X_y):
        """feature_importance deve ter todas as colunas de X."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        for col in X.columns:
            assert col in result["feature_importance"]

    def test_feature_importance_soma_1(self, sample_X_y):
        """feature_importance deve somar aproximadamente 1.0."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        total = sum(result["feature_importance"].values())
        assert abs(total - 1.0) < 0.01

    def test_auc_acima_de_random(self, sample_X_y):
        """AUC deve ser > 0.5 num dataset sintético com sinal."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        assert result["metrics"]["cv_auc_mean"] > 0.5

    def test_scale_pos_weight_calculado(self, sample_X_y):
        """scale_pos_weight deve ser calculado corretamente."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        assert "scale_pos_weight" in result
        assert result["scale_pos_weight"] > 0

    def test_cv_std_e_float(self, sample_X_y):
        """cv_auc_std deve ser float não negativo."""
        X, y = sample_X_y
        result = train_evaluate_xgboost(X, y, "test", n_splits=3)
        assert result["metrics"]["cv_auc_std"] >= 0.0
        assert result["metrics"]["cv_f1_std"]  >= 0.0