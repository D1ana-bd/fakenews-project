# Sistema Inteligente de Deteção de Desinformação em Redes Sociais

Projeto Final de Licenciatura - Ciência de Dados Aplicada (UCP)

## Descrição
Sistema que combina NLP (BERT) e análise de redes para detetar fake news.

## Objetivos
1. Classificar conteúdo como fake/real (F1 > 0.75)
2. Analisar padrões de propagação em redes sociais
3. Integrar ambas as abordagens

## Datasets
- FakeNewsNet (BuzzFeed + PolitiFact + GossipCop)
- LIAR dataset
- CoAID (COVID-19 misinformation)

## Como usar

### Instalação
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Treino
```bash
python src/train_bert.py
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

## Estrutura
```
├── data/          # Dados
├── notebooks/     # Análises exploratórias
├── src/           # Código principal
├── models/        # Modelos treinados
├── results/       # Resultados
└── dashboard/     # Interface Streamlit
```

## Autora
Diana Dória - Universidade Católica Portuguesa (2026)