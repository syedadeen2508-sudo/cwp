"""
Streamlit UI – Cricket Win Prediction Dashboard
Run:  streamlit run app.py
"""

import os
import sys
import pickle
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏏 Cricket Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans",
    "Lucknow Super Giants",
]

VENUES = [
    "Wankhede Stadium", "MA Chidambaram Stadium", "M. Chinnaswamy Stadium",
    "Eden Gardens", "Arun Jaitley Stadium", "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium", "Rajiv Gandhi Stadium", "Narendra Modi Stadium",
    "BRSABV Ekana Cricket Stadium",
]

TEAM_COLORS = {
    "Mumbai Indians":               "#004BA0",
    "Chennai Super Kings":          "#FDB913",
    "Royal Challengers Bangalore":  "#EC1C24",
    "Kolkata Knight Riders":        "#3A225D",
    "Delhi Capitals":               "#17479E",
    "Punjab Kings":                 "#ED1B24",
    "Rajasthan Royals":             "#EA1A85",
    "Sunrisers Hyderabad":          "#F7A721",
    "Gujarat Titans":               "#1C1C1C",
    "Lucknow Super Giants":         "#A72056",
}

TEAM_LOGOS = {t: "🏏" for t in TEAMS}   # placeholder emoji; swap with real URLs if available

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_metadata.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "cricket_dataset.csv")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ── Hero Banner ─────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b4332 50%, #0d1b2a 100%);
    border-radius: 18px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid #2d6a4f;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero h1 { font-size: 2.8rem; font-weight: 900; color: #ffffff; margin: 0; }
.hero p  { font-size: 1.1rem; color: #95d5b2; margin-top: 0.5rem; }

/* ── Metric Cards ────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2rem; font-weight: 700; color: #e0fbfc; }
.metric-label { font-size: 0.78rem; color: #a8dadc; text-transform: uppercase;
                letter-spacing: 1px; margin-top: 0.2rem; }

/* ── Team Badges ─────────────────────────────────────────── */
.team-badge {
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
}

/* ── Win Bar ─────────────────────────────────────────────── */
.win-bar-container {
    background: #1a1a2e;
    border-radius: 50px;
    height: 28px;
    width: 100%;
    overflow: hidden;
    border: 1px solid #333;
    margin: 0.4rem 0;
}
.win-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: white;
}

/* ── Section Headers ─────────────────────────────────────── */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #52b788;
    border-bottom: 2px solid #2d6a4f;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
}

/* ── Prediction Box ──────────────────────────────────────── */
.prediction-win {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    border: 2px solid #52b788;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.prediction-loss {
    background: linear-gradient(135deg, #3d0000, #6b0f0f);
    border: 2px solid #e63946;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.prediction-close {
    background: linear-gradient(135deg, #1a1a00, #4a4a00);
    border: 2px solid #ffd60a;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.prediction-text {
    font-size: 1.6rem;
    font-weight: 900;
    color: #fff;
    margin: 0;
}
.prediction-sub {
    font-size: 0.9rem;
    color: #ccc;
    margin-top: 0.5rem;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b4332 100%);
}
[data-testid="stSidebar"] * { color: #e0fbfc !important; }

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1a1a2e;
    border-radius: 8px 8px 0 0;
    color: #a8dadc;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #2d6a4f !important;
    color: #fff !important;
}

/* ── Spinner overlay ─────────────────────────────────────── */
.stSpinner > div { border-top-color: #52b788 !important; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        HELPER FUNCTIONS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    """Load trained model & metadata (cached across sessions)."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return model, meta


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the full dataset for EDA (cached)."""
    return pd.read_csv(DATA_PATH)


def model_trained() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)


def run_prediction(model, venue, batting_team, bowling_team,
                   toss_winner, toss_decision,
                   first_innings_score, first_innings_wickets, target,
                   balls_bowled, runs_scored, wickets_fallen) -> dict:
    """Compute derived features and run model inference."""
    overs_completed  = balls_bowled // 6
    balls_in_over    = balls_bowled % 6
    runs_needed      = max(0, target - runs_scored)
    balls_left       = max(0, 120 - balls_bowled)
    crr = round(runs_scored / (balls_bowled / 6), 2) if balls_bowled > 0 else 0.0
    rrr = round(runs_needed / (balls_left  / 6), 2) if balls_left  > 0 else 999.0

    row = pd.DataFrame([{
        "venue": venue, "batting_team": batting_team, "bowling_team": bowling_team,
        "toss_winner": toss_winner, "toss_decision": toss_decision,
        "first_innings_score": first_innings_score,
        "first_innings_wickets": first_innings_wickets,
        "target": target, "balls_bowled": balls_bowled,
        "overs_completed": overs_completed, "balls_in_over": balls_in_over,
        "runs_scored": runs_scored, "wickets_fallen": wickets_fallen,
        "runs_needed": runs_needed, "balls_left": balls_left,
        "crr": crr, "rrr": rrr,
    }])

    win_prob = float(model.predict_proba(row)[0][1])
    label    = int(model.predict(row)[0])

    return {
        "win_probability":   round(win_prob, 4),
        "lose_probability":  round(1 - win_prob, 4),
        "batting_team_wins": label,
        "crr": crr, "rrr": rrr,
        "runs_needed": runs_needed, "balls_left": balls_left,
        "overs_completed": overs_completed,
    }


def gauge_chart(prob: float, team: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={"text": f"{team}<br><span style='font-size:0.8em;color:#aaa'>Win Probability</span>",
               "font": {"size": 16, "color": "#fff"}},
        number={"suffix": "%", "font": {"size": 36, "color": "#fff"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#aaa",
                     "tickfont": {"color": "#aaa"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0,  33], "color": "#3d0000"},
                {"range": [33, 60], "color": "#4a4a00"},
                {"range": [60, 100], "color": "#1b4332"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 4},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        font={"color": "#fff"}, height=260, margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


def win_prob_bar(bat_team: str, bowl_team: str,
                 bat_color: str, bowl_color: str,
                 bat_prob: float) -> go.Figure:
    bowl_prob = 1 - bat_prob
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[bat_prob * 100], y=["Win %"],
        orientation="h", name=bat_team,
        marker_color=bat_color,
        text=[f"{bat_prob*100:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[bowl_prob * 100], y=["Win %"],
        orientation="h", name=bowl_team,
        marker_color=bowl_color,
        text=[f"{bowl_prob*100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack", paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        font={"color": "#fff"}, height=110,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def simulate_over_by_over(model, base_params: dict, batting_team: str) -> pd.DataFrame:
    """Simulate win probability over 120 balls for a dramatic chart."""
    records = []
    for b in range(6, 121, 6):
        params = dict(base_params)
        # Interpolate run-scoring (simple linear + noise)
        frac = b / 120
        target = params["target"]
        runs_so_far = int(target * frac * (0.85 + 0.3 * np.random.random()))
        wkts = min(10, int(b / 12 * np.random.uniform(0.5, 1.8)))

        res = run_prediction(
            model,
            params["venue"], params["batting_team"], params["bowling_team"],
            params["toss_winner"], params["toss_decision"],
            params["first_innings_score"], params["first_innings_wickets"],
            target, b, runs_so_far, wkts,
        )
        records.append({
            "Over": b // 6,
            "Win Probability (%)": round(res["win_probability"] * 100, 2),
            "Runs Scored": runs_so_far,
            "Wickets": wkts,
        })
    return pd.DataFrame(records)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          SIDEBAR                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 🏏 Cricket Win Predictor")
    st.markdown("---")

    # Model status
    if model_trained():
        st.success("✅ Model Ready")
    else:
        st.warning("⚠️ Model not trained yet")

    st.markdown("### 🔧 Train / Retrain Model")
    n_samples = st.slider("Training Samples", 2000, 10000, 6000, 500)

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training ML models … this takes ~30s"):
            try:
                # Generate dataset first
                gen_script = os.path.join(BASE_DIR, "data", "generate_dataset.py")
                import importlib.util, types

                # Run dataset generation inline
                spec = importlib.util.spec_from_file_location("gen", gen_script)
                gen_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gen_mod)
                df = gen_mod.generate_dataset(n_samples)
                df.to_csv(DATA_PATH, index=False)
                st.session_state["dataset_refreshed"] = True

                # Train
                import train_model
                importlib.reload(train_model)
                train_model.train(DATA_PATH)

                # Clear caches
                load_model_and_meta.clear()
                load_dataset.clear()
                st.success(f"🎉 Model trained on {n_samples} samples!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
- **Task**: Binary classification  
- **Target**: Will chasing team win?  
- **Features**: 17 match features  
- **Models**: RF, GB, ET, LR  
- **Best**: Selected by AUC-ROC  
    """)
    st.markdown("---")
    st.caption("Made with ❤️ using Scikit-Learn + Streamlit")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           HERO BANNER                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.markdown("""
<div class="hero">
  <h1>🏏 Cricket Win Predictor</h1>
  <p>Real-time IPL / T20 match win probability powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ── Guard: model must be trained ──────────────────────────────────────────────
if not model_trained():
    st.info("👈 **Please train the model first** using the sidebar button.")
    st.stop()

model, meta = load_model_and_meta()

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              TABS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

tab_predict, tab_live, tab_eda, tab_model = st.tabs([
    "🎯 Predict", "📡 Live Simulator", "📊 Data Explorer", "🤖 Model Insights"
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<p class="section-header">⚙️ Match Setup</p>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        batting_team = st.selectbox("🏏 Batting Team (Chasing)", TEAMS, key="bt")
    with col_m2:
        bowling_team = st.selectbox("🎳 Bowling Team (Defending)", TEAMS,
                                    index=1, key="bwt")

    if batting_team == bowling_team:
        st.error("⚠️ Batting and bowling teams must be different!")
        st.stop()

    col_v, col_tw, col_td = st.columns(3)
    with col_v:
        venue = st.selectbox("🏟️ Venue", VENUES, key="venue")
    with col_tw:
        toss_winner = st.selectbox("🪙 Toss Winner", [batting_team, bowling_team], key="tw")
    with col_td:
        toss_decision = st.selectbox("📋 Toss Decision", ["bat", "field"], key="td")

    st.markdown("---")
    st.markdown('<p class="section-header">🏟️ First Innings</p>', unsafe_allow_html=True)

    col_fi1, col_fi2, col_fi3 = st.columns(3)
    with col_fi1:
        first_innings_score = st.number_input("Score", 50, 300, 175, key="fis")
    with col_fi2:
        first_innings_wickets = st.number_input("Wickets Lost", 0, 10, 7, key="fiw")
    with col_fi3:
        target = st.number_input("Target (Runs to Chase)", 51, 301,
                                  int(first_innings_score) + 1, key="tgt")

    st.markdown("---")
    st.markdown('<p class="section-header">⏱️ Current Match State (2nd Innings)</p>',
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        balls_bowled = st.slider("Balls Bowled", 6, 119, 60, key="bb")
        overs_done = balls_bowled // 6
        balls_extra = balls_bowled % 6
        st.caption(f"📍 Over {overs_done}.{balls_extra}  ({120 - balls_bowled} balls left)")
    with col_s2:
        runs_scored = st.number_input("Runs Scored So Far", 0,
                                       int(target) - 1, min(int(target)//2, 80), key="rs")
    with col_s3:
        wickets_fallen = st.number_input("Wickets Fallen", 0, 10, 3, key="wf")

    # ── Predict Button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Win Probability", use_container_width=True, type="primary")

    if predict_btn:
        if int(runs_scored) >= int(target):
            st.success(f"✅ {batting_team} has already won! They reached the target.")
        else:
            with st.spinner("🔄 Running ML model …"):
                result = run_prediction(
                    model, venue, batting_team, bowling_team,
                    toss_winner, toss_decision,
                    int(first_innings_score), int(first_innings_wickets), int(target),
                    int(balls_bowled), int(runs_scored), int(wickets_fallen),
                )

            win_p  = result["win_probability"]
            lose_p = result["lose_probability"]
            bat_c  = TEAM_COLORS.get(batting_team, "#52b788")
            bowl_c = TEAM_COLORS.get(bowling_team, "#e63946")

            # ── Prediction verdict ────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<p class="section-header">🔮 Prediction Result</p>',
                        unsafe_allow_html=True)

            if win_p >= 0.65:
                box_cls = "prediction-win"
                icon = "🟢"
                verdict = f"{batting_team} is LIKELY TO WIN"
            elif win_p <= 0.35:
                box_cls = "prediction-loss"
                icon = "🔴"
                verdict = f"{bowling_team} is LIKELY TO WIN"
            else:
                box_cls = "prediction-close"
                icon = "🟡"
                verdict = "IT'S A CLOSE MATCH!"

            st.markdown(f"""
            <div class="{box_cls}">
              <p class="prediction-text">{icon} {verdict}</p>
              <p class="prediction-sub">
                {batting_team}: <b>{win_p*100:.1f}%</b> &nbsp;|&nbsp;
                {bowling_team}: <b>{lose_p*100:.1f}%</b>
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauges ────────────────────────────────────────────────────────
            gc1, gc2 = st.columns(2)
            with gc1:
                st.plotly_chart(gauge_chart(win_p, batting_team, bat_c),
                                use_container_width=True)
            with gc2:
                st.plotly_chart(gauge_chart(lose_p, bowling_team, bowl_c),
                                use_container_width=True)

            # ── Win-share bar ─────────────────────────────────────────────────
            st.plotly_chart(
                win_prob_bar(batting_team, bowling_team, bat_c, bowl_c, win_p),
                use_container_width=True,
            )

            # ── Live match stats ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<p class="section-header">📈 Live Match Stats</p>',
                        unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            stats = [
                (result["runs_needed"], "Runs Needed"),
                (result["balls_left"],  "Balls Left"),
                (f"{result['crr']:.2f}", "Current RR"),
                (f"{result['rrr']:.2f}", "Required RR"),
                (f"{result['rrr'] - result['crr']:.2f}", "RR Gap"),
            ]
            for col, (val, lbl) in zip([m1, m2, m3, m4, m5], stats):
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{val}</div>
                  <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            # Store for live simulator
            st.session_state["last_prediction"] = {
                "venue": venue, "batting_team": batting_team,
                "bowling_team": bowling_team, "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "first_innings_score": first_innings_score,
                "first_innings_wickets": first_innings_wickets,
                "target": target,
            }


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – LIVE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<p class="section-header">📡 Live Over-by-Over Simulator</p>',
                unsafe_allow_html=True)
    st.info("Configure a match scenario below and watch how the win probability evolves ball-by-ball across all 20 overs.")

    lc1, lc2 = st.columns(2)
    with lc1:
        sim_bat  = st.selectbox("Batting Team", TEAMS, key="sim_bt")
        sim_bowl = st.selectbox("Bowling Team", TEAMS, index=1, key="sim_bw")
        sim_target = st.number_input("Target", 100, 280, 170, key="sim_tgt")
    with lc2:
        sim_venue  = st.selectbox("Venue", VENUES, key="sim_v")
        sim_toss_w = st.selectbox("Toss Winner", [sim_bat, sim_bowl]
                                   if sim_bat != sim_bowl else TEAMS[:2], key="sim_tw")
        sim_toss_d = st.selectbox("Toss Decision", ["bat", "field"], key="sim_td")
        sim_fi_score = st.number_input("1st Innings Score", 80, 280,
                                        int(sim_target) - 1, key="sim_fis")

    if sim_bat == sim_bowl:
        st.error("⚠️ Teams must differ!")
    else:
        if st.button("▶️ Run Simulation", use_container_width=True, type="primary"):
            with st.spinner("Simulating 20 overs …"):
                sim_params = {
                    "venue": sim_venue, "batting_team": sim_bat,
                    "bowling_team": sim_bowl, "toss_winner": sim_toss_w,
                    "toss_decision": sim_toss_d,
                    "first_innings_score": sim_fi_score,
                    "first_innings_wickets": 7,
                    "target": sim_target,
                }
                sim_df = simulate_over_by_over(model, sim_params, sim_bat)
                st.session_state["sim_df"] = sim_df

        if "sim_df" in st.session_state:
            sim_df = st.session_state["sim_df"]
            bat_c  = TEAM_COLORS.get(sim_bat, "#52b788")

            # Win probability line chart
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=sim_df["Over"], y=sim_df["Win Probability (%)"],
                mode="lines+markers",
                line=dict(color=bat_c, width=3),
                marker=dict(size=8, color=bat_c, line=dict(color="#fff", width=1)),
                fill="tozeroy",
                fillcolor=f"rgba({int(bat_c[1:3],16)},{int(bat_c[3:5],16)},{int(bat_c[5:7],16)},0.15)",
                name=f"{sim_bat} Win %",
            ))
            fig_sim.add_hline(y=50, line_dash="dash", line_color="#ffd60a",
                              annotation_text="50% (Even)", annotation_font_color="#ffd60a")
            fig_sim.update_layout(
                title=dict(text=f"Win Probability — {sim_bat} vs {sim_bowl}",
                           font=dict(color="#fff", size=16)),
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
                font=dict(color="#fff"),
                xaxis=dict(title="Over", gridcolor="#2d2d2d", range=[1, 20]),
                yaxis=dict(title="Win Probability (%)", gridcolor="#2d2d2d",
                           range=[0, 100]),
                height=420, margin=dict(l=40, r=20, t=50, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Runs & Wickets chart
            fig_rw = go.Figure()
            fig_rw.add_trace(go.Bar(
                x=sim_df["Over"], y=sim_df["Runs Scored"],
                name="Runs Scored", marker_color="#52b788", opacity=0.8,
            ))
            fig_rw.add_trace(go.Scatter(
                x=sim_df["Over"], y=sim_df["Wickets"] * 20,
                mode="lines+markers", name="Wickets ×20",
                line=dict(color="#e63946", width=2),
                marker=dict(size=7),
                yaxis="y2",
            ))
            fig_rw.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
                font=dict(color="#fff"),
                title=dict(text="Runs Scored & Wickets by Over",
                           font=dict(color="#fff")),
                xaxis=dict(title="Over", gridcolor="#2d2d2d"),
                yaxis=dict(title="Runs", gridcolor="#2d2d2d"),
                yaxis2=dict(title="Wickets", overlaying="y", side="right",
                            range=[0, 10]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=350, margin=dict(l=40, r=60, t=50, b=40),
            )
            st.plotly_chart(fig_rw, use_container_width=True)

            st.dataframe(
                sim_df.style.format({"Win Probability (%)": "{:.1f}%"})
                    .background_gradient(subset=["Win Probability (%)"],
                                         cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – DATA EXPLORER (EDA)
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<p class="section-header">📊 Dataset Explorer</p>',
                unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.warning("No dataset found. Train the model first.")
    else:
        df = load_dataset()

        # KPI row
        e1, e2, e3, e4 = st.columns(4)
        kpis = [
            (f"{len(df):,}", "Total Snapshots"),
            (f"{df['batting_team'].nunique()}", "Teams"),
            (f"{df['venue'].nunique()}", "Venues"),
            (f"{df['batting_team_wins'].mean()*100:.1f}%", "Chaser Win Rate"),
        ]
        for col, (val, lbl) in zip([e1, e2, e3, e4], kpis):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)

        # Win rate by team
        with ec1:
            wr = (df.groupby("batting_team")["batting_team_wins"]
                    .mean().reset_index()
                    .sort_values("batting_team_wins", ascending=True))
            wr["color"] = wr["batting_team"].map(TEAM_COLORS).fillna("#52b788")
            fig_wr = go.Figure(go.Bar(
                x=wr["batting_team_wins"] * 100,
                y=wr["batting_team"],
                orientation="h",
                marker_color=wr["color"].tolist(),
                text=[f"{v:.1f}%" for v in wr["batting_team_wins"]*100],
                textposition="outside",
            ))
            fig_wr.update_layout(
                title="Win Rate When Chasing (by Team)",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
                font=dict(color="#fff"), height=380,
                xaxis=dict(title="Win %", gridcolor="#2d2d2d"),
                yaxis=dict(title=""),
                margin=dict(l=10, r=80, t=50, b=30),
            )
            st.plotly_chart(fig_wr, use_container_width=True)

        # Target distribution
        with ec2:
            fig_tgt = px.histogram(
                df, x="target", nbins=40,
                title="Distribution of Targets",
                color_discrete_sequence=["#52b788"],
            )
            fig_tgt.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
                font=dict(color="#fff"), height=380,
                xaxis=dict(title="Target Runs", gridcolor="#2d2d2d"),
                yaxis=dict(title="Frequency", gridcolor="#2d2d2d"),
            )
            st.plotly_chart(fig_tgt, use_container_width=True)

        ec3, ec4 = st.columns(2)

        # CRR vs RRR scatter
        with ec3:
            sample = df.sample(min(800, len(df)), random_state=1)
            fig_rr = px.scatter(
                sample, x="crr", y="rrr",
                color="batting_team_wins",
                color_discrete_map={0: "#e63946", 1: "#52b788"},
                title="CRR vs RRR (coloured by outcome)",
                labels={"crr": "Current Run Rate", "rrr": "Required Run Rate",
                        "batting_team_wins": "Win"},
                opacity=0.7,
            )
            fig_rr.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
                font=dict(color="#fff"), height=380,
            )
            st.plotly_chart(fig_rr, use_container_width=True)

        # Win rate by venue
        with ec4:
            vn = (df.groupby("venue")["batting_team_wins"]
                    .mean().reset_index()
                    .sort_values("batting_team_wins", ascending=True))
            fig_vn = go.Figure(go.Bar(
                x=vn["batting_team_wins"] * 100,
                y=[v[:20] + "…" if len(v) > 20 else v for v in vn["venue"]],
                orientation="h",
                marker_color="#3a86ff",
                text=[f"{v:.1f}%" for v in vn["batting_team_wins"]*100],
                textposition="outside",
            ))
            fig_vn.update_layout(
                title="Chase Win Rate by Venue",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
                font=dict(color="#fff"), height=380,
                xaxis=dict(title="Win %", gridcolor="#2d2d2d"),
                yaxis=dict(title=""),
                margin=dict(l=10, r=80, t=50, b=30),
            )
            st.plotly_chart(fig_vn, use_container_width=True)

        # Raw data preview
        st.markdown("---")
        with st.expander("📋 Raw Dataset Preview"):
            st.dataframe(df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<p class="section-header">🤖 Model Performance & Insights</p>',
                unsafe_allow_html=True)

    # Model comparison bar
    all_res = meta.get("all_results", [])
    if all_res:
        res_df = pd.DataFrame(all_res).sort_values("auc", ascending=True)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            x=res_df["accuracy"] * 100, y=res_df["name"],
            orientation="h", name="Accuracy (%)",
            marker_color="#52b788", text=[f"{v:.1f}%" for v in res_df["accuracy"]*100],
            textposition="outside",
        ))
        fig_cmp.add_trace(go.Bar(
            x=res_df["auc"] * 100, y=res_df["name"],
            orientation="h", name="AUC-ROC (%)",
            marker_color="#3a86ff", text=[f"{v:.1f}%" for v in res_df["auc"]*100],
            textposition="outside",
            visible="legendonly",
        ))
        fig_cmp.update_layout(
            title="Model Comparison — Accuracy & AUC-ROC",
            barmode="group",
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
            font=dict(color="#fff"), height=320,
            xaxis=dict(range=[0, 115], title="%", gridcolor="#2d2d2d"),
            margin=dict(l=10, r=90, t=50, b=30),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Best model KPIs
    mc1, mc2, mc3 = st.columns(3)
    kpis2 = [
        (f"{meta['accuracy']*100:.2f}%", "Best Model Accuracy"),
        (f"{meta['auc']*100:.2f}%", "AUC-ROC Score"),
        (meta["model_name"], "Algorithm"),
    ]
    for col, (val, lbl) in zip([mc1, mc2, mc3], kpis2):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1.6rem">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fi_df = meta.get("feature_importance", pd.DataFrame())
    if not fi_df.empty:
        # Clean OHE prefixes for display
        fi_df = fi_df.copy()
        fi_df["display_name"] = (fi_df["feature"]
            .str.replace(r"^cat__ohe__", "", regex=True)
            .str.replace(r"^num__scaler__", "", regex=True)
            .str.replace("_", " ").str.title())

        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"].head(15),
            y=fi_df["display_name"].head(15),
            orientation="h",
            marker_color=px.colors.sequential.Viridis[:15],
            text=[f"{v:.3f}" for v in fi_df["importance"].head(15)],
            textposition="outside",
        ))
        fig_fi.update_layout(
            title="Top 15 Feature Importances",
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
            font=dict(color="#fff"), height=420,
            xaxis=dict(title="Importance", gridcolor="#2d2d2d"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=90, t=50, b=30),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrix
    cm = meta.get("confusion_matrix")
    if cm is not None:
        labels = ["Bowling Team Wins", "Batting Team Wins"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=labels, y=labels,
            colorscale="Greens",
            text=cm, texttemplate="%{text}",
            showscale=True,
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a1a2e",
            font=dict(color="#fff"), height=350,
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(l=30, r=30, t=50, b=30),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    report = meta.get("report", {})
    if report:
        st.markdown("### 📋 Classification Report")
        rep_rows = []
        for k, v in report.items():
            if isinstance(v, dict):
                rep_rows.append({"Class": k, **{m: round(v[m], 4)
                                                 for m in ["precision","recall","f1-score","support"]}})
        if rep_rows:
            st.dataframe(pd.DataFrame(rep_rows).set_index("Class"),
                         use_container_width=True)

    st.markdown("---")
    st.caption(f"🏆 Best Model: **{meta['model_name']}** | "
               f"Accuracy: **{meta['accuracy']*100:.2f}%** | "
               f"AUC-ROC: **{meta['auc']*100:.2f}%**")
