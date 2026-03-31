"""
build_graphs.py
===============
Módulo de Análise de Redes — Fake News Detection Project
Universidade Católica Portuguesa | Diana Dória

Objetivo: Construir grafos sintéticos de propagação calibrados com dados reais
do FakeNewsNet (GossipCop), calcular métricas estruturais e de comunidades,
e guardar os resultados para integração no ensemble final.

Metodologia:
    Como o FakeNewsNet (versão CSV) apenas disponibiliza tweet_ids e não a
    estrutura completa de retweets, os grafos de propagação são gerados
    sinteticamente usando o modelo Barabási-Albert (scale-free), calibrado
    pelo número real de tweets por artigo. Esta abordagem é academicamente
    válida e amplamente usada na literatura quando os dados de propagação
    completos não estão disponíveis (ver R7 na Gestão de Riscos do relatório).

Estrutura do grafo por artigo:
    - Nós   : utilizadores (tweet_ids tratados como IDs de utilizador)
    - Arestas: relações de disseminação A → B ("A partilhou de B")
    - Nó 0  : fonte original (primeiro tweet / artigo)
    - Tipo  : DiGraph dirigido
"""

import random
import warnings
from pathlib import Path

import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logger ────────────────────────────────────────────────────────────────────
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.FKNWS.utils.get_logger import get_logger

logger = get_logger("build_graphs")

# ── Constantes ────────────────────────────────────────────────────────────────
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "fakenewsnet"
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

# Parâmetros do modelo Barabási-Albert
BA_M = 2          # cada novo nó liga-se a m nós existentes
RANDOM_SEED = 42  # reprodutibilidade
MAX_ARTICLES = 100  # processar amostra (None = todos)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(max_articles: int | None = MAX_ARTICLES) -> pd.DataFrame:
    """
    Carrega os CSVs do GossipCop (fake + real) e devolve um DataFrame unificado
    com colunas: id, label, n_tweets, tweet_ids_list.

    Args:
        max_articles: Número máximo de artigos a carregar (None = todos).
                      Usa amostra estratificada para manter proporção fake/real.

    Returns:
        DataFrame com os artigos e metadados de propagação.
    """
    logger.info("A carregar dados do FakeNewsNet...")

    files = {
        "fake": [
            DATA_RAW / "gossipcop_fake.csv",
            DATA_RAW / "politifact_fake.csv",
        ],
        "real": [
            DATA_RAW / "gossipcop_real.csv",
            DATA_RAW / "politifact_real.csv",
        ],
    }

    dfs = []
    for label, paths in files.items():
        for path in paths:
            if not path.exists():
                logger.warning(f"Ficheiro não encontrado: {path}")
                continue
            df = pd.read_csv(path)
            df["label"] = label
            df["label_binary"] = 1 if label == "fake" else 0
            # Guardar a fonte (gossipcop vs politifact) para análise posterior
            df["source"] = "gossipcop" if "gossipcop" in path.name else "politifact"
            dfs.append(df)
            logger.info(f"  {label} ({path.name}): {len(df)} artigos")

    df_all = pd.concat(dfs, ignore_index=True)

    # Parsear tweet_ids: string separada por \t → lista
    df_all["tweet_ids_list"] = (
        df_all["tweet_ids"]
        .dropna()
        .str.split("\t")
    )
    df_all["n_tweets"] = df_all["tweet_ids_list"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Remover artigos sem tweets (não têm grafo de propagação)
    df_all = df_all[df_all["n_tweets"] > 0].reset_index(drop=True)
    logger.info(f"  Total após filtro (n_tweets > 0): {len(df_all)} artigos")

    # Amostra estratificada (mantém proporção fake/real)
    if max_articles is not None and max_articles < len(df_all):
        df_all = (
            df_all
            .groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), max_articles // 2),
                random_state=RANDOM_SEED
            ))
            .reset_index(drop=True)
        )
        logger.info(f"  Amostra estratificada: {len(df_all)} artigos "
                    f"({df_all['label'].value_counts().to_dict()})")

    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSTRUÇÃO DO GRAFO
# ══════════════════════════════════════════════════════════════════════════════

def build_propagation_graph(n_nodes: int, seed: int = RANDOM_SEED) -> nx.DiGraph:
    """
    Constrói um grafo sintético de propagação usando o modelo Barabási-Albert.

    O modelo BA gera redes scale-free (distribuição de grau em lei de potência),
    característica das redes sociais reais. Nós com mais ligações têm maior
    probabilidade de receber novas ligações (preferential attachment).

    Args:
        n_nodes: Número de nós (= número de tweets do artigo).
        seed: Semente aleatória para reprodutibilidade.

    Returns:
        DiGraph dirigido onde nó 0 é a fonte original.
    """
    if n_nodes < 2:
        # Grafo trivial: apenas a fonte
        G = nx.DiGraph()
        G.add_node(0, role="source")
        return G

    # BA gera grafo não-dirigido; convertemos para dirigido
    # (arestas apontam da fonte para os partilhadores)
    m = min(BA_M, n_nodes - 1)  # m não pode ser >= n_nodes
    G_undirected = nx.barabasi_albert_graph(n_nodes, m, seed=seed)

    # Converter para DiGraph: BFS a partir do nó 0 para definir direção
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())

    # BFS: nó pai → nó filho (direção de propagação)
    visited = set()
    queue = [0]
    visited.add(0)

    while queue:
        node = queue.pop(0)
        for neighbor in G_undirected.neighbors(node):
            if neighbor not in visited:
                G.add_edge(node, neighbor)  # pai → filho
                visited.add(neighbor)
                queue.append(neighbor)

    # Atributos dos nós
    nx.set_node_attributes(G, "user", "type")
    G.nodes[0]["role"] = "source"

    # Timestamps sintéticos (propagação exponencial: cada hop demora minutos/horas)
    rng = random.Random(seed)
    timestamps = {0: 0}  # fonte no tempo 0
    for node in nx.bfs_tree(G, 0).nodes():
        if node != 0:
            parent = next(iter(G.predecessors(node)), 0)
            # Intervalo exponencial entre 5 min e 12 horas (em minutos)
            delta = rng.expovariate(1 / 60)  # média = 60 min
            delta = max(5, min(delta, 720))   # clamp [5, 720] minutos
            timestamps[node] = timestamps.get(parent, 0) + delta

    nx.set_node_attributes(G, timestamps, "timestamp_min")

    return G


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(G: nx.DiGraph, article_id: str, label: str) -> dict:
    """
    Calcula métricas estruturais e temporais de um grafo de propagação.

    Métricas estruturais:
        - degree_centrality_mean/max : centralidade de grau média e máxima
        - betweenness_mean/max       : betweenness centrality (nós "ponte")
        - clustering_mean            : coeficiente de agrupamento médio
        - density                    : densidade da rede
        - n_nodes / n_edges          : dimensão do grafo
        - n_communities              : número de comunidades (Louvain)
        - modularity                 : qualidade das comunidades

    Métricas temporais:
        - propagation_speed          : nós alcançados por hora (Δnós/Δtempo)
        - max_depth                  : profundidade máxima da árvore
        - time_to_peak_min           : tempo até ao nó mais distante (minutos)

    Args:
        G          : Grafo de propagação.
        article_id : ID do artigo (para logging).
        label      : "fake" ou "real".

    Returns:
        Dicionário com todas as métricas calculadas.
    """
    metrics = {"article_id": article_id, "label": label}
    n = G.number_of_nodes()
    metrics["n_nodes"] = n
    metrics["n_edges"] = G.number_of_edges()

    if n < 2:
        # Grafo trivial — métricas não aplicáveis
        for key in ["degree_centrality_mean", "degree_centrality_max",
                    "betweenness_mean", "betweenness_max", "clustering_mean",
                    "density", "n_communities", "modularity",
                    "propagation_speed", "max_depth", "time_to_peak_min"]:
            metrics[key] = 0.0
        return metrics

    # ── Grafo não-dirigido para métricas que o requerem ──────────────────────
    G_und = G.to_undirected()

    # ── Degree Centrality ─────────────────────────────────────────────────────
    deg_cent = nx.degree_centrality(G_und)
    metrics["degree_centrality_mean"] = round(np.mean(list(deg_cent.values())), 6)
    metrics["degree_centrality_max"] = round(max(deg_cent.values()), 6)

    # ── Betweenness Centrality ────────────────────────────────────────────────
    # k=min(n,50): aproximação para grafos grandes (mais rápido)
    k = min(n, 50)
    bet_cent = nx.betweenness_centrality(G_und, k=k, seed=RANDOM_SEED)
    metrics["betweenness_mean"] = round(np.mean(list(bet_cent.values())), 6)
    metrics["betweenness_max"] = round(max(bet_cent.values()), 6)

    # ── Clustering Coefficient ────────────────────────────────────────────────
    metrics["clustering_mean"] = round(nx.average_clustering(G_und), 6)

    # ── Densidade ────────────────────────────────────────────────────────────
    metrics["density"] = round(nx.density(G_und), 6)

    # ── Louvain Community Detection ───────────────────────────────────────────
    try:
        partition = community_louvain.best_partition(G_und, random_state=RANDOM_SEED)
        metrics["n_communities"] = len(set(partition.values()))
        metrics["modularity"] = round(
            community_louvain.modularity(partition, G_und), 6
        )
    except Exception as e:
        logger.warning(f"  Louvain falhou para {article_id}: {e}")
        metrics["n_communities"] = 1
        metrics["modularity"] = 0.0

    # ── Métricas Temporais ────────────────────────────────────────────────────
    timestamps = nx.get_node_attributes(G, "timestamp_min")

    if timestamps:
        max_time = max(timestamps.values()) if timestamps else 0
        metrics["time_to_peak_min"] = round(max_time, 2)

        # Velocidade: nós alcançados por hora
        if max_time > 0:
            metrics["propagation_speed"] = round((n - 1) / (max_time / 60), 4)
        else:
            metrics["propagation_speed"] = 0.0
    else:
        metrics["time_to_peak_min"] = 0.0
        metrics["propagation_speed"] = 0.0

    # ── Profundidade máxima (cascade depth) ───────────────────────────────────
    try:
        metrics["max_depth"] = nx.dag_longest_path_length(G)
    except Exception:
        metrics["max_depth"] = 0

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def build_all_graphs(df: pd.DataFrame) -> tuple[list[dict], dict[str, nx.DiGraph]]:
    """
    Itera sobre todos os artigos, constrói o grafo e calcula as métricas.

    Args:
        df: DataFrame com artigos (output de load_data).

    Returns:
        Tuplo (lista de métricas, dicionário article_id → grafo).
    """
    all_metrics = []
    graphs = {}

    logger.info(f"A construir grafos para {len(df)} artigos...")

    for i, row in df.iterrows():
        article_id = str(row["id"])
        label = row["label"]
        n_nodes = int(row["n_tweets"])

        # Seed única por artigo (reprodutível)
        seed = RANDOM_SEED + i

        # Construir grafo
        G = build_propagation_graph(n_nodes, seed=seed)
        G.graph["article_id"] = article_id
        G.graph["label"] = label
        graphs[article_id] = G

        # Calcular métricas
        m = calculate_metrics(G, article_id, label)
        all_metrics.append(m)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progresso: {i + 1}/{len(df)} artigos processados")

    logger.info(f"Grafos construídos: {len(graphs)} total")
    return all_metrics, graphs


def save_metrics(all_metrics: list[dict]) -> pd.DataFrame:
    """
    Guarda as métricas num CSV e devolve o DataFrame.

    Args:
        all_metrics: Lista de dicionários com métricas por artigo.

    Returns:
        DataFrame com todas as métricas.
    """
    df_metrics = pd.DataFrame(all_metrics)
    output_path = RESULTS_METRICS / "network_metrics.csv"
    df_metrics.to_csv(output_path, index=False)
    logger.info(f"Métricas guardadas em: {output_path}")
    return df_metrics


def summarize_metrics(df_metrics: pd.DataFrame) -> None:
    """
    Imprime um resumo comparativo fake vs real das métricas calculadas.

    Args:
        df_metrics: DataFrame com métricas (output de save_metrics).
    """
    logger.info("=" * 60)
    logger.info("RESUMO DAS MÉTRICAS — FAKE vs REAL")
    logger.info("=" * 60)

    numeric_cols = [
        "n_nodes", "n_edges", "degree_centrality_mean", "degree_centrality_max",
        "betweenness_mean", "betweenness_max", "clustering_mean", "density",
        "n_communities", "modularity", "propagation_speed",
        "max_depth", "time_to_peak_min"
    ]

    summary = (
        df_metrics
        .groupby("label")[numeric_cols]
        .mean()
        .round(4)
        .T
    )
    print("\n", summary.to_string(), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   Análise de Redes — FakeNewsNet         ║")
    logger.info("╚══════════════════════════════════════════╝")

    # 1. Carregar dados
    df = load_data(max_articles=MAX_ARTICLES)

    # 2. Construir grafos + métricas
    all_metrics, graphs = build_all_graphs(df)

    # 3. Guardar métricas
    df_metrics = save_metrics(all_metrics)

    # 4. Resumo
    summarize_metrics(df_metrics)

    logger.info("✓ Pipeline de análise de redes concluído!")