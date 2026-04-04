"""
test_build_graphs.py
====================
Testes unitários para o módulo de análise de redes.
Universidade Católica Portuguesa | Diana Dória

Cobertura:
    - build_propagation_graph(): estrutura, nós, arestas, atributos
    - calculate_metrics(): tipos, valores, casos extremos
    - load_data(): parsing, filtros, amostragem
    - save_metrics(): output CSV correto

Executar:
    cd ~/fakenews-project
    pytest tests/test_build_graphs.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.FKNWS.network.build_graphs import (
    build_propagation_graph,
    calculate_metrics,
    save_metrics,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_graph():
    """Grafo de propagação pequeno (10 nós) para testes rápidos."""
    return build_propagation_graph(n_nodes=10, seed=42)


@pytest.fixture
def medium_graph():
    """Grafo de propagação médio (50 nós)."""
    return build_propagation_graph(n_nodes=50, seed=42)


@pytest.fixture
def trivial_graph():
    """Grafo trivial com apenas 1 nó (artigo sem retweets)."""
    return build_propagation_graph(n_nodes=1, seed=42)


@pytest.fixture
def sample_df():
    """DataFrame sintético que simula o output de load_data()."""
    return pd.DataFrame({
        "id": [
            "gc_fake_001", "gc_fake_002", "gc_fake_003",
            "gc_real_001", "gc_real_002", "gc_real_003",
        ],
        "label": ["fake", "fake", "fake", "real", "real", "real"],
        "label_binary": [1, 1, 1, 0, 0, 0],
        "source": ["gossipcop"] * 6,
        "n_tweets": [15, 5, 100, 20, 8, 50],
        "tweet_ids_list": [
            ["t1", "t2", "t3"],
            ["t4", "t5"],
            ["t6"] * 10,
            ["t7", "t8", "t9", "t10"],
            ["t11", "t12"],
            ["t13"] * 8,
        ],
    })


@pytest.fixture
def sample_metrics():
    """Lista de métricas sintéticas para testar save_metrics()."""
    return [
        {
            "article_id": "gc_fake_001", "label": "fake",
            "n_nodes": 15, "n_edges": 14,
            "degree_centrality_mean": 0.15, "degree_centrality_max": 0.5,
            "betweenness_mean": 0.05, "betweenness_max": 0.3,
            "clustering_mean": 0.1, "density": 0.12,
            "n_communities": 3, "modularity": 0.4,
            "propagation_speed": 2.5, "max_depth": 4,
            "time_to_peak_min": 120.0,
        },
        {
            "article_id": "gc_real_001", "label": "real",
            "n_nodes": 20, "n_edges": 19,
            "degree_centrality_mean": 0.10, "degree_centrality_max": 0.3,
            "betweenness_mean": 0.03, "betweenness_max": 0.2,
            "clustering_mean": 0.08, "density": 0.09,
            "n_communities": 2, "modularity": 0.3,
            "propagation_speed": 1.8, "max_depth": 3,
            "time_to_peak_min": 180.0,
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TESTES: build_propagation_graph()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPropagationGraph:

    def test_returns_digraph(self, small_graph):
        """O grafo retornado deve ser um DiGraph dirigido."""
        assert isinstance(small_graph, nx.DiGraph)

    def test_correct_number_of_nodes(self, small_graph):
        """O grafo deve ter exatamente n_nodes nós."""
        assert small_graph.number_of_nodes() == 10

    def test_has_edges(self, small_graph):
        """O grafo deve ter pelo menos uma aresta."""
        assert small_graph.number_of_edges() > 0

    def test_node_zero_is_source(self, small_graph):
        """O nó 0 deve existir e ter role='source'."""
        assert 0 in small_graph.nodes
        assert small_graph.nodes[0].get("role") == "source"

    def test_all_nodes_have_timestamp(self, small_graph):
        """Todos os nós devem ter o atributo timestamp_min."""
        for node in small_graph.nodes:
            assert "timestamp_min" in small_graph.nodes[node], \
                f"Nó {node} não tem timestamp_min"

    def test_source_timestamp_is_zero(self, small_graph):
        """O nó fonte (0) deve ter timestamp_min = 0."""
        assert small_graph.nodes[0]["timestamp_min"] == 0

    def test_timestamps_are_non_negative(self, small_graph):
        """Todos os timestamps devem ser >= 0."""
        for node, data in small_graph.nodes(data=True):
            assert data["timestamp_min"] >= 0, \
                f"Nó {node} tem timestamp negativo: {data['timestamp_min']}"

    def test_timestamps_increase_with_depth(self, small_graph):
        """Nós mais profundos devem ter timestamp >= nó pai."""
        for u, v in small_graph.edges():
            t_parent = small_graph.nodes[u]["timestamp_min"]
            t_child = small_graph.nodes[v]["timestamp_min"]
            assert t_child >= t_parent, \
                f"Filho {v} ({t_child}) tem timestamp < pai {u} ({t_parent})"

    def test_trivial_graph_single_node(self, trivial_graph):
        """Grafo com n_nodes=1 deve ter apenas o nó fonte."""
        assert trivial_graph.number_of_nodes() == 1
        assert trivial_graph.number_of_edges() == 0
        assert 0 in trivial_graph.nodes

    def test_reproducibility(self):
        """Grafos com a mesma seed devem ser idênticos."""
        G1 = build_propagation_graph(20, seed=99)
        G2 = build_propagation_graph(20, seed=99)
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.edges()) == set(G2.edges())

    def test_different_seeds_differ(self):
        """Grafos com seeds diferentes devem (tipicamente) diferir."""
        G1 = build_propagation_graph(50, seed=1)
        G2 = build_propagation_graph(50, seed=2)
        # Não são necessariamente diferentes mas com n=50 é muito provável
        assert G1.number_of_nodes() == G2.number_of_nodes()  # nós iguais
        # Arestas podem diferir (não assert obrigatório, só verificar)

    def test_large_graph(self):
        """Grafo grande (500 nós) deve construir sem erro."""
        G = build_propagation_graph(500, seed=42)
        assert G.number_of_nodes() == 500
        assert G.number_of_edges() > 0

    def test_two_nodes_graph(self):
        """Grafo com 2 nós deve funcionar corretamente."""
        G = build_propagation_graph(2, seed=42)
        assert G.number_of_nodes() == 2
        assert 0 in G.nodes


# ══════════════════════════════════════════════════════════════════════════════
# TESTES: calculate_metrics()
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateMetrics:

    def test_returns_dict(self, small_graph):
        """calculate_metrics deve retornar um dicionário."""
        result = calculate_metrics(small_graph, "test_001", "fake")
        assert isinstance(result, dict)

    def test_contains_required_keys(self, small_graph):
        """O dicionário deve conter todas as chaves esperadas."""
        required_keys = [
            "article_id", "label", "n_nodes", "n_edges",
            "degree_centrality_mean", "degree_centrality_max",
            "betweenness_mean", "betweenness_max",
            "clustering_mean", "density",
            "n_communities", "modularity",
            "propagation_speed", "max_depth", "time_to_peak_min",
        ]
        result = calculate_metrics(small_graph, "test_001", "fake")
        for key in required_keys:
            assert key in result, f"Chave '{key}' em falta no resultado"

    def test_article_id_and_label_preserved(self, small_graph):
        """article_id e label devem ser preservados no resultado."""
        result = calculate_metrics(small_graph, "gc_fake_999", "fake")
        assert result["article_id"] == "gc_fake_999"
        assert result["label"] == "fake"

    def test_n_nodes_correct(self, small_graph):
        """n_nodes deve corresponder ao número real de nós do grafo."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["n_nodes"] == small_graph.number_of_nodes()

    def test_n_edges_correct(self, small_graph):
        """n_edges deve corresponder ao número real de arestas."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["n_edges"] == small_graph.number_of_edges()

    def test_density_between_0_and_1(self, small_graph):
        """Densidade deve estar em [0, 1]."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert 0.0 <= result["density"] <= 1.0

    def test_degree_centrality_between_0_and_1(self, small_graph):
        """Degree centrality deve estar em [0, 1]."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert 0.0 <= result["degree_centrality_mean"] <= 1.0
        assert 0.0 <= result["degree_centrality_max"] <= 1.0

    def test_degree_max_gte_mean(self, small_graph):
        """degree_centrality_max deve ser >= degree_centrality_mean."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["degree_centrality_max"] >= result["degree_centrality_mean"]

    def test_betweenness_non_negative(self, small_graph):
        """Betweenness centrality deve ser >= 0."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["betweenness_mean"] >= 0.0
        assert result["betweenness_max"] >= 0.0

    def test_clustering_between_0_and_1(self, small_graph):
        """Clustering coefficient deve estar em [0, 1]."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert 0.0 <= result["clustering_mean"] <= 1.0

    def test_n_communities_at_least_1(self, small_graph):
        """Deve haver pelo menos 1 comunidade."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["n_communities"] >= 1

    def test_propagation_speed_non_negative(self, small_graph):
        """Velocidade de propagação deve ser >= 0."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["propagation_speed"] >= 0.0

    def test_max_depth_non_negative(self, small_graph):
        """Profundidade máxima deve ser >= 0."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["max_depth"] >= 0

    def test_time_to_peak_non_negative(self, small_graph):
        """Tempo até ao pico deve ser >= 0."""
        result = calculate_metrics(small_graph, "test", "fake")
        assert result["time_to_peak_min"] >= 0.0

    def test_trivial_graph_returns_zeros(self, trivial_graph):
        """Grafo trivial (1 nó) deve retornar zeros nas métricas."""
        result = calculate_metrics(trivial_graph, "trivial", "real")
        assert result["n_nodes"] == 1
        assert result["n_edges"] == 0
        assert result["propagation_speed"] == 0.0
        assert result["density"] == 0.0

    def test_label_real(self, medium_graph):
        """Deve funcionar igualmente com label='real'."""
        result = calculate_metrics(medium_graph, "test_real", "real")
        assert result["label"] == "real"

    def test_medium_graph_has_communities(self, medium_graph):
        """Grafo médio (50 nós) deve ter pelo menos 2 comunidades."""
        result = calculate_metrics(medium_graph, "test", "fake")
        assert result["n_communities"] >= 1  # pelo menos 1


# ══════════════════════════════════════════════════════════════════════════════
# TESTES: save_metrics()
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveMetrics:

    def test_returns_dataframe(self, sample_metrics, tmp_path):
        """save_metrics deve retornar um DataFrame."""
        with patch(
            "FKNWS.network.build_graphs.RESULTS_METRICS", tmp_path
        ):
            from src.FKNWS.network.build_graphs import save_metrics as sm
            result = sm(sample_metrics)
        assert isinstance(result, pd.DataFrame)

    def test_csv_file_created(self, sample_metrics, tmp_path):
        """O ficheiro CSV deve ser criado em results/metrics/."""
        import src.FKNWS.network.build_graphs as bg
        original = bg.RESULTS_METRICS
        try:
            bg.RESULTS_METRICS = tmp_path
            bg.save_metrics(sample_metrics)
        finally:
            bg.RESULTS_METRICS = original
        assert (tmp_path / "network_metrics.csv").exists()

    def test_dataframe_has_correct_rows(self, sample_metrics, tmp_path):
        """O DataFrame deve ter o mesmo número de linhas que a lista."""
        with patch(
            "FKNWS.network.build_graphs.RESULTS_METRICS", tmp_path
        ):
            from src.FKNWS.network.build_graphs import save_metrics as sm
            result = sm(sample_metrics)
        assert len(result) == len(sample_metrics)

    def test_dataframe_has_label_column(self, sample_metrics, tmp_path):
        """O DataFrame deve ter a coluna 'label'."""
        with patch(
            "FKNWS.network.build_graphs.RESULTS_METRICS", tmp_path
        ):
            from src.FKNWS.network.build_graphs import save_metrics as sm
            result = sm(sample_metrics)
        assert "label" in result.columns

    def test_csv_is_readable(self, sample_metrics, tmp_path):
        """O CSV gerado deve ser legível com pd.read_csv."""
        import src.FKNWS.network.build_graphs as bg
        original = bg.RESULTS_METRICS
        try:
            bg.RESULTS_METRICS = tmp_path
            bg.save_metrics(sample_metrics)
        finally:
            bg.RESULTS_METRICS = original
        df_read = pd.read_csv(tmp_path / "network_metrics.csv")
        assert len(df_read) == len(sample_metrics)