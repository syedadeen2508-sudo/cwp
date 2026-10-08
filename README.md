# 🏏 Cricket Win Prediction — End-to-End ML Project

Real-time IPL / T20 cricket win probability prediction using Machine Learning.

## 📁 Project Structure

```
CWP/
├── app.py                    # Streamlit interactive UI (4 tabs)
├── train_model.py            # ML training pipeline (4 models)
├── requirements.txt
├── .streamlit/
│   └── config.toml           # Dark cricket theme
├── data/
│   ├── generate_dataset.py   # Synthetic IPL data generator
│   └── cricket_dataset.csv   # Generated (after training)
├── models/
│   ├── best_model.pkl        # Best trained model (after training)
│   └── model_metadata.pkl    # Metrics, confusion matrix, feature importance
└── utils/
    └── predictor.py          # Inference utility module
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Random Forest | 200 trees, max_depth=15 |
| Gradient Boosting | 200 estimators, lr=0.05 |
| Extra Trees | 200 trees, max_depth=15 |
| Logistic Regression | C=1.0, max_iter=1000 |

Best model is selected by **AUC-ROC** and saved to `models/best_model.pkl`.

---

## 🎯 Features Used

| Feature | Description |
|---|---|
| `batting_team` | Team currently batting (chasing) |
| `bowling_team` | Team currently bowling (defending) |
| `venue` | Match stadium |
| `toss_winner` / `toss_decision` | Toss info |
| `first_innings_score` | Score set by defending team |
| `target` | Runs needed to win |
| `balls_bowled` | Progress in 2nd innings |
| `runs_scored` | Runs scored so far |
| `wickets_fallen` | Wickets lost |
| `crr` | Current Run Rate |
| `rrr` | Required Run Rate |
| `runs_needed` | Remaining runs |
| `balls_left` | Remaining balls |

---

## 🖥️ UI Tabs

| Tab | Description |
|---|---|
| 🎯 **Predict** | Enter match state → get win probability with gauges |
| 📡 **Live Simulator** | Simulate over-by-over win probability chart |
| 📊 **Data Explorer** | EDA charts: win rates, target distributions, scatter plots |
| 🤖 **Model Insights** | Feature importances, confusion matrix, classification report |

---

## 🏟️ Teams Supported (IPL)

Mumbai Indians, Chennai Super Kings, Royal Challengers Bangalore,
Kolkata Knight Riders, Delhi Capitals, Punjab Kings,
Rajasthan Royals, Sunrisers Hyderabad, Gujarat Titans, Lucknow Super Giants

---

## 📊 Dataset

- **8,000 synthetic IPL-style T20 match snapshots**
- Each row = one live game state during the 2nd innings
- Features engineered from realistic team-strength heuristics
- Balanced classes (~50/50 win/lose)
