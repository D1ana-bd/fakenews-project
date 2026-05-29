# Sistema Inteligente de Deteção de Desinformação em Redes Sociais

Projeto Final de Licenciatura — Ciência de Dados Aplicada | Universidade Católica Portuguesa (2026)

## Descrição

Sistema que combina **Processamento de Linguagem Natural** (BERT) e **Análise de Redes**
para detetar automaticamente fake news e caracterizar os seus padrões de propagação
em redes sociais.

## Objetivos

| Objetivo | Descrição | Estado    |
|----------|-----------|-----------|
| 1 — NLP | Classificar conteúdo como fake/real com BERT fine-tuned | Concluído |
| 2 — Redes | Analisar padrões de propagação com NetworkX + Louvain | Concluído |
| 3 — Integração | Combinar NLP e redes num alpha score ponderado | Concluído |

## Resultados

### Objetivo 1 — NLP
| Modelo | F1 (teste) |
|--------|-----------|
| Baseline TF-IDF + Logistic Regression | 0.40 |
| **BERT Fine-tuned (cascata LIAR → FakeNewsNet)** | **0.52** |
| Melhoria | +30.4% |

### Objetivo 2 — Redes (21.693 artigos)
| Métrica | Fake | Real |
|---------|------|------|
| Degree Centrality (média) | 0.204 | 0.075 |
| Betweenness (média) | 0.112 | 0.059 |
| Modularidade | 0.431 | 0.576 |
| Velocidade Propagação | 14.9 | 10.4 |
| Super-spreaders (% fake) | **70.4%** | 25.5% baseline |

## Datasets

| Dataset | Tamanho | Uso |
|---------|---------|-----|
| FakeNewsNet (BuzzFeed + PolitiFact + GossipCop) | 422 artigos | Treino NLP + Grafos |
| LIAR | 12.791 statements | Pré-fine-tune BERT |
| CoAID | 4.189 artigos | Validação externa |

## Instalação
```bash
git clone https://github.com/[username]/fakenews-project.git
cd fakenews-project

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Como Usar

### Pré-processamento
```bash
python src/FKNWS/data_prep/data_preprocessing.py
```

### Treino BERT (Objetivo 1)
```bash
# Treino completo (cascata LIAR → FakeNewsNet)
python src/FKNWS/models/train_bert.py --phase all

# Com batch size reduzido (GPU com pouca memória)
python src/FKNWS/models/train_bert.py --phase all --batch_size 8

# Testar pipeline com subset pequeno
python src/FKNWS/models/train_bert.py --phase all --test_run
```

### Análise de Redes (Objetivo 2)
```bash
python src/FKNWS/network/build_graphs.py
```

### Dashboard (Objetivo 3)
```bash
streamlit run dashboard/app.py
```

### Testes
```bash
pytest tests/ -v
```

## Estrutura do Projeto
fakenews-project/
├── data/
│   ├── raw/                    # Dados originais (não versionados)
│   │   ├── fakenewsnet/
│   │   ├── LIAR/
│   │   └── CoAID/
│   └── processed/              # Dados processados (não versionados)
│       └── fakenewsnet/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_nlp_results.ipynb
│   └── 04_network_analysis.ipynb (em progresso)
├── src/FKNWS/
│   ├── data_prep/
│   │   └── data_preprocessing.py
│   ├── models/
│   │   ├── train_bert.py
│   │   └── README.md
│   ├── network/
│   │   └── build_graphs.py
│   └── utils/
│       └── get_logger.py
├── models/                     # Pesos do modelo (não versionados)
│   ├── bert_liar/
│   └── bert_fake_news/
├── results/
│   ├── figures/                # Gráficos gerados
│   └── metrics/                # Métricas em JSON
├── tests/
│   ├── test_data_preprocessing.py
│   └── test_train_bert.py
├── .gitignore
├── requirements.txt
└── README.md

## Arquitetura do Sistema
FakeNewsNet ──→ Baseline TF-IDF + LR  ──→ F1 = 0.40
│
└──→ BERT Fine-tune (Fase 2) ──→ F1 = 0.52
↑
LIAR ────→ BERT Pré-fine-tune (Fase 1)
GossipCop ──→ Grafos NetworkX ──→ Métricas de Rede
│
┌─────────┘
↓
Alpha Score = α × NLP + (1-α) × Rede
│
↓
Dashboard Streamlit (?)

## Autora

**Diana Dória**
Licenciatura em Ciência de Dados Aplicada
Universidade Católica Portuguesa — Braga, 2026