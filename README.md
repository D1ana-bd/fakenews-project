# Sistema Inteligente de Deteção de Desinformação em Redes Sociais

**Projeto Individual — Licenciatura em Ciência de Dados Aplicada**  
**Universidade Católica Portuguesa — Braga**  
**Autor:** Diana Dória | **Ano letivo:** 2025/2026

---

## Descrição

Este projeto desenvolve um sistema modular de deteção de desinformação que combina **Processamento de Linguagem Natural (NLP)** com **Análise de Redes Sociais**, integrados num sistema de ensemble. O sistema foi avaliado sobre datasets públicos do FakeNewsNet e atingiu AUC=0.8287 no sistema final (XGBoost + Zero-shot).

### Objetivos
1. **Módulo NLP** — Classificação textual de notícias como fake/real usando BERT em cascata
2. **Módulo Redes** — Identificação de padrões estruturais de propagação em grafos sintéticos Barabási-Albert
3. **Sistema Ensemble** — Integração de ambos os módulos num classificador XGBoost

---

## Resultados Principais

| Modelo | Métrica | Valor |
|--------|---------|-------|
| Baseline TF-IDF | F1 | 0.40 |
| BERT Cascata (LIAR → FakeNewsNet) | F1 | 0.52 (+30.4%) |
| Zero-shot (cross-encoder/nli-deberta-v3-small) | F1 | 0.6349 |
| Módulo Redes isolado | AUC | 0.674 |
| Ensemble Linear (α=0.5) | AUC | 0.672 |
| **XGBoost + Zero-shot (Final)** | **AUC** | **0.8287** |

**Feature importance XGBoost:**
- `degree_centrality_mean`: 55.35%
- `modularity`: 19.65%
- `betweenness_mean`: 9.61%
- `propagation_speed`: 6.38%
- `score_nlp_zeroshot`: 5.72%
- `n_communities`: 3.29%

---

## Estrutura do Repositório

```
fakenews-project/
│
├── data/
│   ├── raw/
│   │   ├── fakenewsnet/          # GossipCop CSVs (fake + real)
│   │   └── LIAR/                 # Dataset LIAR (train/valid/test .tsv)
│   └── processed/
│       └── fakenewsnet/          # train.csv / val.csv / test.csv
│
├── src/
│   └── FKNWS/
│       ├── models/
│       │   ├── train_bert.py     # Fine-tuning BERT em cascata
│       │   └── zero_shot.py      # Classificador zero-shot
│       ├── network/
│       │   └── build_graphs.py   # Construção grafos BA + métricas
│       ├── integration/
│       │   ├── integration.py    # Ensemble linear
│       │   └── ensemble_xgboost.py # Ensemble XGBoost
│       └── utils/
│           └── get_logger.py     # Logger
│
├── models/                       # Modelos treinados (BERT, DistilBERT)
│
├── results/
│   ├── metrics/                  # JSONs e CSVs com resultados
│   │   ├── results_nlp.json
│   │   ├── results_baseline.json
│   │   ├── results_xgboost.json
│   │   ├── results_zeroshot.json
│   │   ├── network_metrics.csv
│   │   └── integration_comparison.csv
│   └── figures/                  # Figuras geradas
│
├── tests/                        # Testes unitários
│   ├── test_bert.py
│   ├── test_xgboost.py
│   └── test_zero_shot.py
│
├── dashboard/
│   └── app.py                    # Dashboard Streamlit
│
├── notebooks/                    # Análise exploratória
│
├── requirements.txt
└── README.md
```

---

## Instalação

### Pré-requisitos
- Python 3.10+
- conda (recomendado) ou venv
- GPU com CUDA (recomendado para treino BERT)

### Configuração do ambiente

```bash
# Clonar o repositório
git clone https://github.com/<username>/fakenews-project.git
cd fakenews-project

# Criar ambiente conda
conda create -n fakenews python=3.10
conda activate fakenews

# Instalar dependências
pip install -r requirements.txt
```

---

## Utilização

### 1. Treino do Módulo NLP (BERT em Cascata)

```bash
# Fase 1 — Pré-treino no LIAR
python src/FKNWS/models/train_bert.py --phase 1

# Fase 2 — Fine-tuning no FakeNewsNet
python src/FKNWS/models/train_bert.py --phase 2

# Ambas as fases em sequência
python src/FKNWS/models/train_bert.py --phase all

# Teste rápido com 10% dos dados
python src/FKNWS/models/train_bert.py --phase all --test_run
```

### 2. Classificação Zero-shot

```bash
python src/FKNWS/models/zero_shot.py
```

### 3. Construção dos Grafos e Métricas de Rede

```bash
python src/FKNWS/network/build_graphs.py
```

Gera `results/metrics/network_metrics.csv` com 9 métricas por artigo.

### 4. Ensemble Linear

```bash
# Testar todos os alphas (0.5, 0.6, 0.7, 0.8)
python src/FKNWS/integration/integration.py

# Alpha específico
python src/FKNWS/integration/integration.py --alpha 0.5
```

### 5. Ensemble XGBoost (Sistema Final)

```bash
python src/FKNWS/integration/ensemble_xgboost.py
```

Gera `results/metrics/results_xgboost.json` com resultados por validação cruzada (5 folds).

### 6. Dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

---

## Datasets

| Dataset | Utilização | Exemplos |
|---------|-----------|----------|
| **LIAR** | Pré-treino NLP (Fase 1) | 10.240 |
| **FakeNewsNet BuzzFeed/PolitiFact** | Fine-tuning NLP (Fase 2) | 422 (336 treino / 43 val / 43 teste) | 
| **GossipCop** | Análise de redes + Ensemble | 21.695 artigos |
| **CoAID** | Validação exploratória | — | 

> **Nota:** Os datasets não estão incluídos no repositório. O FakeNewsNet está disponível no [Kaggle](https://www.kaggle.com/datasets/mdepak/fakenewsnet). O LIAR está disponivel no [Kaggle](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset).

---

## Limitações Conhecidas

- **Grafos sintéticos:** Os grafos de propagação são gerados com o modelo Barabási-Albert calibrado pelo número de tweets, não por estrutura real de retweets (não disponível no GossipCop CSV).
- **Mismatch treino/inferência NLP:** O BERT foi treinado com título + texto completo mas a inferência no ensemble opera apenas sobre títulos (GossipCop).
- **Volume de dados NLP reduzido:** 336 exemplos de treino limitam a generalização do classificador supervisionado.
- **Língua:** Todos os datasets são em inglês.

---

## Hiperparâmetros BERT

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | `bert-base-uncased` |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Épocas | 3 |
| Early stopping (patience) | 2 |
| Optimizer | AdamW |
| LR Scheduler | Linear warmup (10% steps) |
| Max sequence length | 512 tokens |

---

## Stack Tecnológico

- **ML/NLP:** PyTorch, HuggingFace Transformers, Scikit-learn
- **Redes:** NetworkX, python-louvain
- **Ensemble:** XGBoost
- **Visualização:** Matplotlib, Seaborn, Pyvis
- **Dashboard:** Streamlit
- **Versão Python:** 3.10
- **GPU:** NVIDIA (CUDA) — recomendado para treino BERT

---

## Referências

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Shu et al. (2020). FakeNewsNet: A Data Repository with News Content, Social Context and Spatio-temporal Information.
- Wang (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.
- Vosoughi et al. (2018). The spread of true and false news online. Science.
- Barabási & Albert (1999). Emergence of scaling in random networks. Science.

