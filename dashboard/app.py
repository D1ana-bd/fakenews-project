"""
app.py
======
Dashboard interativa — Sistema Inteligente de Deteção de Desinformação
Diana Dória — Universidade Católica Portuguesa, 2026

Uso:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

# ── paths ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
MODEL_PATH  = ROOT / "models" / "bert_fake_news"

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeNews Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0f0f1a !important;
    font-family: 'Outfit', sans-serif !important;
    color: #e8e8f0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.3) !important;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

[data-testid="stMain"] { background: #0f0f1a !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(108,99,255,0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetricLabel"] { color: #a0a0c0 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108,99,255,0.6) !important;
}

[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
}

hr { border-color: rgba(108,99,255,0.2) !important; margin: 24px 0 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1a2e; }
::-webkit-scrollbar-thumb { background: #6c63ff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def card(content: str, padding: str = "24px") -> str:
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 20px;
        padding: {padding};
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    ">{content}</div>
    """

def gradient_title(text: str, size: str = "2.4rem") -> str:
    return f"""
    <h1 style="
        font-family: 'Outfit', sans-serif;
        font-size: {size};
        font-weight: 800;
        background: linear-gradient(135deg, #6c63ff 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        line-height: 1.2;
    ">{text}</h1>
    """

def badge(text: str, color: str) -> str:
    return f"""
    <span style="
        background: {color}22;
        color: {color};
        border: 1px solid {color}55;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
    ">{text}</span>
    """

def result_card_fake(score: float) -> str:
    pct = int(score * 100)
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 2px solid rgba(239,68,68,0.5);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    ">
        <div style="font-size: 4rem; margin-bottom: 8px;">🔴</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444; margin-bottom: 8px;">FAKE NEWS</div>
        <div style="font-size: 1.1rem; color: #fca5a5; margin-bottom: 20px;">Confiança: <strong>{pct}%</strong></div>
        <div style="background: rgba(239,68,68,0.2); border-radius: 10px; height: 12px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #ef4444, #f97316); height: 100%; width: {pct}%; border-radius: 10px;"></div>
        </div>
    </div>
    """

def result_card_real(score: float) -> str:
    pct = int((1 - score) * 100)
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        border: 2px solid rgba(34,197,94,0.5);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    ">
        <div style="font-size: 4rem; margin-bottom: 8px;">🟢</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #22c55e; margin-bottom: 8px;">REAL NEWS</div>
        <div style="font-size: 1.1rem; color: #86efac; margin-bottom: 20px;">Confiança: <strong>{pct}%</strong></div>
        <div style="background: rgba(34,197,94,0.2); border-radius: 10px; height: 12px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #22c55e, #84cc16); height: 100%; width: {pct}%; border-radius: 10px;"></div>
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="A carregar modelo BERT...")
def load_bert():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_data(show_spinner=False)
def load_data():
    with open(METRICS_DIR / "results_integration.json") as f:
        integration = json.load(f)
    with open(METRICS_DIR / "results_xgboost.json") as f:
        xgboost = json.load(f)
    comparison = pd.read_csv(METRICS_DIR / "integration_comparison.csv")
    with open(METRICS_DIR / "predictions_bert.json") as f:
        preds = json.load(f)
    df_preds = pd.DataFrame(preds["predictions"])
    df_net   = pd.read_csv(METRICS_DIR / "network_metrics.csv")
    df_net   = df_net.drop(columns=["label"], errors="ignore")
    df       = df_preds.merge(df_net, on="article_id", how="inner")
    return integration, xgboost, comparison, df


def predict(text: str) -> float:
    model, tokenizer, device = load_bert()
    enc = tokenizer(
        text, truncation=True, max_length=512,
        return_tensors="pt", padding=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = F.softmax(logits, dim=-1)
    return probs[0, 1].item()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(gradient_title("🔍 FakeNews<br>Detector", "1.6rem"), unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; font-size:12px; margin-top:-8px;'>Diana Dória · UCP · 2026</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navegação",
        ["🏠  Início", "🔍  Classificador", "🕸️  Explorador de Grafos", "📊  Resultados"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#a0a0c0; font-size:11px; line-height:1.8;'>
        <b style='color:#6c63ff;'>Stack técnico</b><br>
        PyTorch · BERT · XGBoost<br>
        NetworkX · Streamlit · Pyvis<br><br>
        <b style='color:#6c63ff;'>Datasets</b><br>
        FakeNewsNet · LIAR<br><br>
        <b style='color:#6c63ff;'>Testes</b><br>
        170+ testes unitários ✅
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 0 — INÍCIO
# ═══════════════════════════════════════════════════════════════════════════

if "Início" in page:
    st.markdown(gradient_title("Sistema Inteligente de Deteção<br>de Desinformação em Redes Sociais", "2.6rem"), unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; font-size:1.05rem; margin-bottom:32px;'>Universidade Católica Portuguesa · Licenciatura em Ciência de Dados Aplicada · 2026</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Artigos Analisados", "21.693", "gossipcop + politifact")
    with c2: st.metric("F1 BERT (Obj. 1)", "0.52", "+30.4% vs baseline")
    with c3: st.metric("Super-spreaders Fake", "70.4%", "vs 25.5% baseline")
    with c4: st.metric("AUC XGBoost", "0.829", "melhor resultado ⭐")

    st.markdown("---")
    st.markdown("<h3 style='color:#e8e8f0; font-weight:700; margin-bottom:20px;'>Objetivos do Projeto</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(card(f"""
        <div style='margin-bottom:12px;'>{badge("Objetivo 1", "#6c63ff")} {badge("✅ Concluído", "#22c55e")}</div>
        <div style='font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:10px;'>🧠 Classificação NLP</div>
        <div style='color:#a0a0c0; font-size:0.9rem; line-height:1.7;'>
            Fine-tuning BERT em cascata:<br>
            pré-treino LIAR → fine-tune FakeNewsNet<br><br>
            <span style='color:#6c63ff; font-weight:600;'>Baseline TF-IDF:</span> F1 = 0.40<br>
            <span style='color:#a855f7; font-weight:600;'>BERT Cascata:</span> F1 = 0.52 (+30.4%)<br>
            <span style='color:#ec4899; font-weight:600;'>Zero-shot:</span> F1 = 0.635 ⭐
        </div>
        """), unsafe_allow_html=True)

    with col2:
        st.markdown(card(f"""
        <div style='margin-bottom:12px;'>{badge("Objetivo 2", "#f59e0b")} {badge("✅ Concluído", "#22c55e")}</div>
        <div style='font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:10px;'>🕸️ Análise de Redes</div>
        <div style='color:#a0a0c0; font-size:0.9rem; line-height:1.7;'>
            Grafos Barabási-Albert calibrados<br>
            por tweet counts reais.<br><br>
            <span style='color:#f59e0b; font-weight:600;'>Super-spreaders fake:</span> 70.4%<br>
            <span style='color:#f59e0b; font-weight:600;'>Velocidade fake:</span> +43.4%<br>
            <span style='color:#ec4899; font-weight:600;'>AUC rede isolada:</span> 0.674
        </div>
        """), unsafe_allow_html=True)

    with col3:
        st.markdown(card(f"""
        <div style='margin-bottom:12px;'>{badge("Objetivo 3", "#ec4899")} {badge("✅ Concluído", "#22c55e")}</div>
        <div style='font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:10px;'>⚡ Ensemble XGBoost</div>
        <div style='color:#a0a0c0; font-size:0.9rem; line-height:1.7;'>
            XGBoost aprende pesos ótimos<br>
            NLP + features de rede.<br><br>
            <span style='color:#ec4899; font-weight:600;'>Ensemble linear:</span> AUC=0.672<br>
            <span style='color:#ec4899; font-weight:600;'>XGBoost Zero-shot:</span> AUC=0.829 ⭐<br>
            <span style='color:#ec4899; font-weight:600;'>degree_centrality:</span> 55-63% importance
        </div>
        """), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#e8e8f0; font-weight:700; margin-bottom:20px;'>Arquitetura do Sistema</h3>", unsafe_allow_html=True)
    st.markdown(card("""
    <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; text-align:center;'>
        <div style='flex:1; min-width:80px;'>
            <div style='font-size:2rem;'>📰</div>
            <div style='color:#6c63ff; font-weight:600; font-size:0.8rem; margin-top:6px;'>FakeNewsNet<br>LIAR</div>
        </div>
        <div style='color:#a0a0c0; font-size:1.5rem;'>→</div>
        <div style='flex:1; min-width:80px;'>
            <div style='font-size:2rem;'>⚙️</div>
            <div style='color:#a855f7; font-weight:600; font-size:0.8rem; margin-top:6px;'>Pré-processamento<br>& Grafos BA</div>
        </div>
        <div style='color:#a0a0c0; font-size:1.5rem;'>→</div>
        <div style='flex:1; min-width:80px;'>
            <div style='font-size:2rem;'>🧠</div>
            <div style='color:#ec4899; font-weight:600; font-size:0.8rem; margin-top:6px;'>BERT<br>Fine-tuned</div>
        </div>
        <div style='color:#a0a0c0; font-size:1.2rem;'>+</div>
        <div style='flex:1; min-width:80px;'>
            <div style='font-size:2rem;'>🕸️</div>
            <div style='color:#f59e0b; font-weight:600; font-size:0.8rem; margin-top:6px;'>Network<br>Features</div>
        </div>
        <div style='color:#a0a0c0; font-size:1.5rem;'>→</div>
        <div style='flex:1; min-width:80px;'>
            <div style='font-size:2rem;'>⚡</div>
            <div style='color:#22c55e; font-weight:600; font-size:0.8rem; margin-top:6px;'>XGBoost<br>Ensemble</div>
        </div>
        <div style='color:#a0a0c0; font-size:1.5rem;'>→</div>
        <div style='flex:1; min-width:80px;'>
            <div style='font-size:2rem;'>🔍</div>
            <div style='color:#22c55e; font-weight:600; font-size:0.8rem; margin-top:6px;'>FAKE / REAL<br>+ Confiança</div>
        </div>
    </div>
    """), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CLASSIFICADOR
# ═══════════════════════════════════════════════════════════════════════════

elif "Classificador" in page:
    st.markdown(gradient_title("🔍 Classificador ao Vivo"), unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-bottom:28px;'>Insere o título de uma notícia e o modelo BERT classifica-a em tempo real.</p>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("<label style='color:#a0a0c0; font-size:13px; text-transform:uppercase; letter-spacing:1px;'>Título da notícia</label>", unsafe_allow_html=True)
        user_text = st.text_area(
            label="",
            placeholder="Ex: Scientists discover new treatment that cures cancer overnight...",
            height=160,
            label_visibility="collapsed",
            key="user_input",
        )

        st.markdown("<p style='color:#a0a0c0; font-size:12px; margin-top:12px;'>Ou experimenta um exemplo:</p>", unsafe_allow_html=True)
        ex_col1, ex_col2, ex_col3 = st.columns(3)

        with ex_col1:
            if st.button("Exemplo Fake 🔴", use_container_width=True):
                st.session_state["classify_text"] = "SHOCKING: Government hiding cure for cancer, whistleblower reveals massive conspiracy"
                st.rerun()
        with ex_col2:
            if st.button("Exemplo Real 🟢", use_container_width=True):
                st.session_state["classify_text"] = "Federal Reserve holds interest rates steady amid economic uncertainty"
                st.rerun()
        with ex_col3:
            if st.button("Exemplo Ambíguo ⚠️", use_container_width=True):
                st.session_state["classify_text"] = "New study suggests coffee may have unexpected health benefits"
                st.rerun()

        # usar texto do session_state se existir
        if "classify_text" in st.session_state and not user_text:
            user_text = st.session_state["classify_text"]

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        classify_btn = st.button("🔍 Classificar", use_container_width=True)

    with col_result:
        text_to_classify = user_text.strip() if user_text else st.session_state.get("classify_text", "")

        if classify_btn and text_to_classify:
            with st.spinner("A analisar..."):
                score = predict(text_to_classify)

            if score >= 0.5:
                st.markdown(result_card_fake(score), unsafe_allow_html=True)
            else:
                st.markdown(result_card_real(score), unsafe_allow_html=True)

            st.markdown(card(f"""
            <div style='font-size:13px; color:#a0a0c0; line-height:1.8;'>
                <b style='color:#e8e8f0;'>Detalhes da predição</b><br><br>
                <span style='color:#6c63ff;'>Score Fake:</span> {score:.4f}<br>
                <span style='color:#6c63ff;'>Score Real:</span> {1-score:.4f}<br>
                <span style='color:#6c63ff;'>Modelo:</span> BERT fine-tuned (cascata)<br>
                <span style='color:#6c63ff;'>Threshold:</span> 0.5<br>
                <span style='color:#6c63ff;'>Input:</span> título only
            </div>
            """), unsafe_allow_html=True)

        elif classify_btn:
            st.warning("Por favor insere um texto para classificar.")
        else:
            st.markdown(card("""
            <div style='text-align:center; padding:40px 0; color:#a0a0c0;'>
                <div style='font-size:3rem; margin-bottom:12px;'>💬</div>
                <div style='font-size:1rem;'>O resultado aparece aqui<br>após a classificação.</div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(card("""
    <div style='font-size:12px; color:#a0a0c0; line-height:1.7;'>
        ⚠️ <b style='color:#f59e0b;'>Nota metodológica:</b>
        O modelo foi treinado com título + texto completo (FakeNewsNet + LIAR).
        A classificação aqui usa apenas o título, pelo que a confiança pode ser inferior ao F1=0.52 obtido no Objetivo 1.
    </div>
    """), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — EXPLORADOR DE GRAFOS
# ═══════════════════════════════════════════════════════════════════════════

elif "Grafos" in page:
    st.markdown(gradient_title("🕸️ Explorador de Grafos"), unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-bottom:28px;'>Explora os grafos de propagação de artigos do dataset.</p>", unsafe_allow_html=True)

    try:
        _, _, _, df = load_data()

        col_ctrl, col_info = st.columns([1, 2], gap="large")

        with col_ctrl:
            label_filter = st.selectbox("Filtrar por", ["Todos", "Fake", "Real"])
            df_filtered = df.copy()
            if label_filter == "Fake":
                df_filtered = df[df["label"] == 1]
            elif label_filter == "Real":
                df_filtered = df[df["label"] == 0]

            sample       = df_filtered.sample(min(100, len(df_filtered)), random_state=42)
            selected     = st.selectbox("Artigo", sample["article_id"].tolist())
            row          = df[df["article_id"] == selected].iloc[0]
            true_label   = "🔴 Fake" if row["label"] == 1 else "🟢 Real"
            pred_label   = "🔴 Fake" if row["pred_nlp"] == 1 else "🟢 Real"

            st.markdown(card(f"""
            <div style='font-size:13px; line-height:2;'>
                <b style='color:#e8e8f0;'>{selected}</b><br>
                <span style='color:#a0a0c0;'>Label real:</span> {true_label}<br>
                <span style='color:#a0a0c0;'>Pred. NLP:</span> {pred_label}<br>
                <span style='color:#a0a0c0;'>Score NLP:</span>
                <span style='color:#6c63ff; font-weight:600;'>{row['score_nlp']:.4f}</span>
            </div>
            """), unsafe_allow_html=True)

        with col_info:
            st.markdown("<h4 style='color:#e8e8f0; margin-bottom:16px;'>Métricas de Rede</h4>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Degree Centrality", f"{row['degree_centrality_mean']:.4f}")
                st.metric("Nº Comunidades", f"{int(row['n_communities'])}")
            with m2:
                st.metric("Betweenness", f"{row['betweenness_mean']:.4f}")
                st.metric("Modularidade", f"{row['modularity']:.4f}")
            with m3:
                st.metric("Vel. Propagação", f"{row['propagation_speed']:.2f}")
                st.metric("Max Depth", f"{int(row['max_depth'])}")

            fake_mean_deg = df[df["label"]==1]["degree_centrality_mean"].mean()
            real_mean_deg = df[df["label"]==0]["degree_centrality_mean"].mean()
            fake_mean_spd = df[df["label"]==1]["propagation_speed"].mean()
            real_mean_spd = df[df["label"]==0]["propagation_speed"].mean()

            st.markdown(card(f"""
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; font-size:13px; text-align:center;'>
                <div>
                    <div style='color:#a0a0c0; margin-bottom:4px;'>Métrica</div>
                    <div style='font-weight:600; color:#e8e8f0;'>Degree</div>
                    <div style='font-weight:600; color:#e8e8f0;'>Velocidade</div>
                </div>
                <div>
                    <div style='color:#ef4444; margin-bottom:4px;'>Média Fake</div>
                    <div style='color:#ef4444; font-weight:700;'>{fake_mean_deg:.4f}</div>
                    <div style='color:#ef4444; font-weight:700;'>{fake_mean_spd:.2f}</div>
                </div>
                <div>
                    <div style='color:#22c55e; margin-bottom:4px;'>Média Real</div>
                    <div style='color:#22c55e; font-weight:700;'>{real_mean_deg:.4f}</div>
                    <div style='color:#22c55e; font-weight:700;'>{real_mean_spd:.2f}</div>
                </div>
            </div>
            """), unsafe_allow_html=True)

        # grafo Pyvis
        st.markdown("---")
        st.markdown("<h4 style='color:#e8e8f0; margin-bottom:8px;'>Grafo de Propagação (sintético BA)</h4>", unsafe_allow_html=True)

        try:
            import networkx as nx
            from pyvis.network import Network
            import tempfile, os

            n_nodes  = min(int(row.get("n_nodes", 30)), 80)
            is_fake  = row["label"] == 1
            G        = nx.barabasi_albert_graph(n_nodes, 2, seed=42)

            net = Network(height="420px", width="100%",
                          bgcolor="#0f0f1a", font_color="#e8e8f0", directed=True)
            net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)

            node_color   = "#ef4444" if is_fake else "#22c55e"
            source_color = "#f97316" if is_fake else "#84cc16"

            for node in G.nodes():
                size  = 25 if node == 0 else max(8, 20 - G.degree(node))
                color = source_color if node == 0 else node_color
                net.add_node(node, size=size, color=color,
                             title=f"Nó {node} | Degree: {G.degree(node)}", label="")

            for u, v in G.edges():
                net.add_edge(u, v, color="rgba(108,99,255,0.3)", width=1)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                net.save_graph(f.name)
                html_content = open(f.name).read()
                os.unlink(f.name)

            st.components.v1.html(html_content, height=440)

        except ImportError:
            st.info("Pyvis não instalado. Corre: `pip install pyvis`")

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

elif "Resultados" in page:
    st.markdown(gradient_title("📊 Resultados & Comparação"), unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0c0; margin-bottom:28px;'>Comparação de todas as abordagens desenvolvidas.</p>", unsafe_allow_html=True)

    try:
        integration, xgboost_res, comparison, df = load_data()
        best_alpha = integration["best_alpha"]
        best       = integration["all_results"][str(best_alpha)]

        # métricas topo
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("AUC Rede isolada",    f"{best['network_only']['auc']:.4f}")
        with c2: st.metric("AUC NLP isolado",     f"{best['nlp_only']['auc']:.4f}")
        with c3: st.metric("AUC Ensemble linear", f"{best['ensemble']['auc']:.4f}")
        with c4: st.metric("AUC XGBoost",         "0.8287", "Zero-shot + rede ⭐")
        with c5: st.metric("Super-spreaders",     "70.4%",  "vs 25.5% baseline")

        st.markdown("---")

        # tabela ensemble linear
        st.markdown("<h4 style='color:#e8e8f0; margin-bottom:16px;'>Ensemble Linear — Todas as configurações α</h4>", unsafe_allow_html=True)
        st.dataframe(
            comparison.style.format(
                {c: "{:.4f}" for c in ["accuracy","precision","recall","f1","auc"]}
            ).highlight_max(
                subset=["accuracy","precision","recall","f1","auc"],
                color="#1a3a2a"
            ).set_properties(**{
                "background-color": "#1a1a2e",
                "color": "#e8e8f0",
                "border": "1px solid rgba(108,99,255,0.2)",
            }),
            use_container_width=True
        )

        st.markdown("---")

        # tabela XGBoost
        st.markdown("<h4 style='color:#e8e8f0; margin-bottom:16px;'>Ensemble XGBoost — 4 Abordagens</h4>", unsafe_allow_html=True)
        xgb_rows = []
        labels_map = {
            "abordagem_1_bert":        "BERT cascata + XGBoost",
            "abordagem_2_zeroshot":    "Zero-shot + XGBoost ⭐",
            "abordagem_3a_distilbert": "DistilBERT títulos + XGBoost",
            "abordagem_3b_bert_titles":"BERT títulos + XGBoost",
        }
        for key, label in labels_map.items():
            if key in xgboost_res["results"]:
                m = xgboost_res["results"][key]["metrics"]
                xgb_rows.append({
                    "Abordagem": label,
                    "CV AUC":    m["cv_auc_mean"],
                    "CV AUC std": m["cv_auc_std"],
                    "CV F1":     m["cv_f1_mean"],
                    "CV F1 std": m["cv_f1_std"],
                })
        df_xgb = pd.DataFrame(xgb_rows)
        st.dataframe(
            df_xgb.style.format({
                "CV AUC": "{:.4f}", "CV AUC std": "{:.4f}",
                "CV F1":  "{:.4f}", "CV F1 std":  "{:.4f}",
            }).highlight_max(subset=["CV AUC","CV F1"], color="#1a3a2a")
            .set_properties(**{
                "background-color": "#1a1a2e",
                "color": "#e8e8f0",
                "border": "1px solid rgba(108,99,255,0.2)",
            }),
            use_container_width=True
        )

        st.markdown("---")

        # figuras
        st.markdown("<h4 style='color:#e8e8f0; margin-bottom:16px;'>Visualizações</h4>", unsafe_allow_html=True)

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            p = FIGURES_DIR / "xgboost_comparison.png"
            if p.exists():
                st.image(str(p), caption="XGBoost — comparação AUC e F1", use_container_width=True)
        with fig_col2:
            p = FIGURES_DIR / "xgboost_feature_importance.png"
            if p.exists():
                st.image(str(p), caption="Feature importance por abordagem", use_container_width=True)

        p = FIGURES_DIR / "super_spreaders.png"
        if p.exists():
            st.markdown("---")
            st.markdown("<h4 style='color:#e8e8f0; margin-bottom:16px;'>Super-spreaders (Objetivo 2)</h4>", unsafe_allow_html=True)
            st.image(str(p), caption="70.4% fake vs 25.5% baseline", use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao carregar resultados: {e}")