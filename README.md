# Deep RL for Personalized News Recommendation
### IST 597 Final Project · Penn State University · Spring 2025

**Authors:** Yogeshvar Reddy Kallam · Ajay Krishna Devulapally  
**Course:** IST 597 — Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-MINDsmall-lightgrey.svg)](https://msnews.github.io)
[![Tianshou](https://img.shields.io/badge/Framework-Tianshou-purple.svg)](https://tianshou.readthedocs.io)

---

## 📌 Abstract

Personalized news recommendation is a challenging sequential decision-making problem where user interests shift dynamically over time. We frame it as a **Markov Decision Process** and compare five RL algorithms — from classical tabular methods to a curiosity-driven deep RL agent.

Using the **MINDsmall dataset** (50K users, 157K sessions, 51K articles), our best model — **DQN with Intrinsic Curiosity Module (ICM)** — achieves:

| Metric | Value |
|--------|-------|
| **HitRate@10** | **0.624** (18× better than tabular MC) |
| **NDCG@10** | **0.443** |
| vs. Standard DQN HitRate@10 | 0.325 → **+92% gain** |

---

## 🗂️ Repository Structure

```
deep-rl-news-recommendation/
│
├── README.md
├── Deep_RL_News_Recommendation_Report.pdf   ← Full research report
│
├── src/
│   ├── data_preprocessing.py    ← MINDsmall data loading & feature engineering
│   ├── state_representation.py  ← Sentence embedding state encoder
│   ├── tabular_agents.py        ← MC Control, Q-Learning, SARSA
│   ├── dqn_agent.py             ← Standard DQN (Tianshou)
│   ├── dqn_icm_agent.py         ← DQN + Intrinsic Curiosity Module
│   └── evaluate.py              ← HitRate@K, NDCG@K, MRR@K metrics
│
└── notebooks/
    └── experiments.ipynb        ← Full experiment notebook
```

---

## 🧠 Problem Formulation (MDP)

News recommendation is modeled as a sequential decision problem:

```
State  (S): User reading history encoded as dense embeddings
             Tabular: category of last clicked article (~15 states)
             DQN:     concat(embed(last 10 titles)) → 3,840-dim vector

Action (A): Select an article from the current impression list
             Tabular: any seen news_id
             DQN+ICM: position in impression list (0–19)

Reward (R): Extrinsic: +1 if user clicked, 0 otherwise
             Intrinsic (ICM only): forward model prediction error
             Total: r = r_ext + β · r_int

Transition: Determined by logged user interaction sequences (offline)

Discount γ: 0.90 (tabular)  |  0.99 (DQN)
```

---

## 📐 Architecture

### Tabular Methods

```
MC Control:    Q(s,a) ← Q(s,a) + 1/N · (Gt − Q(s,a))     [every-visit]
Q-Learning:    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]
SARSA:         Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]

State space:   ~15 categories  (severely limits representation)
```

### Standard DQN

```
Input: 3,840-dim state vector
       (384-dim sentence embedding × last 10 clicked titles)
       
QNet (MLP):
  Linear(3840 → 512) → ReLU
  Linear(512  → 256) → ReLU
  Linear(256  → 20)  → Q-values for each impression slot

Key components:
  ✦ Experience Replay   buffer size = 20,000
  ✦ Target Network      update every 500 steps
  ✦ ε-greedy            1.0 → 0.1 over 20 episodes
  ✦ Adam optimizer      lr = 1e-4, batch = 64
```

### DQN + Intrinsic Curiosity Module (ICM) — Best Model

```
┌─────────────────────────────────────────────────────────────┐
│                     ICM MODULE                               │
│                                                              │
│  s  ──► Encoder φ ──► φ(s)                                   │
│  s' ──► Encoder φ ──► φ(s')                                  │
│                                                              │
│  Inverse Model:  [φ(s), φ(s')] ──► â_t    (predict action)  │
│  Loss_inv = CrossEntropy(â_t, a_t)                           │
│                                                              │
│  Forward Model:  [φ(s), a_t] ──► φ̂(s')   (predict next)    │
│  Loss_fwd = ‖φ̂(s') − φ(s')‖²                               │
│                                                              │
│  Intrinsic Reward: r_i = β · ‖φ̂(s') − φ(s')‖²             │
└─────────────────────────────────────────────────────────────┘
                          ↓
           r_total = r_extrinsic + r_intrinsic
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    DQN (enhanced)                            │
│  Input: [3840-dim state ‖ 20-dim action mask]                │
│  QNet: Linear(3860→512)→ReLU→Linear(512→256)→ReLU→L(256→20) │
│  Prioritized Replay  α=0.6, β=0.4, size=50,000              │
│  Target update: every 320 steps                              │
│  ε-decay: 1.0 → 0.05 over 20,000 steps                      │
│  Adam lr=1e-3, batch=256                                     │
│                                                              │
│  ICM loss weights:                                           │
│    reward_scale = 10.0                                       │
│    inverse_loss = 1.0                                        │
│    forward_loss = 50.0                                       │
└─────────────────────────────────────────────────────────────┘

Framework: Custom DQNPolicyWithICM built on Tianshou
```

---

## 📊 Results

### Performance Comparison (K=10)

| Algorithm | State Space | HitRate@10 | NDCG@10 | MRR@10 |
|-----------|-------------|------------|---------|--------|
| Monte Carlo (MC) | ~15 categories | 0.0347 | 0.0545 | — |
| Q-Learning | ~15 categories | 0.0581 | 0.0300 | — |
| SARSA | ~15 categories | 0.0645 | 0.0321 | — |
| DQN (Standard) | 3,840-dim embed | 0.3248 | 0.2786 | — |
| **DQN + ICM** | **3,840-dim embed** | **0.6240** | **0.4432** | **best** |

### Key Findings

1. **State representation is the dominant factor** — DQN with embeddings outperforms tabular methods by ~10× despite similar algorithms, because 15 category states cannot distinguish user preferences.

2. **Curiosity-driven exploration is highly effective for sparse rewards** — In news recommendation, clicks are rare (sparse reward). ICM's intrinsic reward drives the agent to explore novel user-article combinations, discovering a substantially better policy.

3. **Tabular TD methods marginally outperform MC** — SARSA > Q-Learning > MC on HitRate (TD's bootstrapping gives more frequent updates), but all are bottlenecked by the coarse state space.

---

## 📦 Dataset

**MINDsmall** (Microsoft News Dataset)

| File | Contents |
|------|----------|
| `news.tsv` | ~51,000 articles — ID, category, subcategory, title, abstract |
| `behaviors.tsv` | ~157,000 user sessions — user_id, timestamp, click history, impressions |

Download from: [https://msnews.github.io](https://msnews.github.io)

```
behaviors.tsv format:
  impression_id | user_id | timestamp | history | impressions
  
  history:     "N1234 N5678 N9012 ..."    (previously clicked)
  impressions: "N1111-1 N2222-0 ..."      (shown: 1=clicked, 0=not)
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yogeshvarreddykallam/deep-rl-news-recommendation.git
cd deep-rl-news-recommendation

# Install dependencies
pip install torch tianshou sentence-transformers gymnasium numpy pandas tqdm

# Download MINDsmall dataset from https://msnews.github.io
# Place news.tsv and behaviors.tsv in data/

# Run experiments
jupyter notebook notebooks/experiments.ipynb
```

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural networks (QNet, ICM encoder/models) |
| `tianshou` | DQN policy, replay buffer, trainer |
| `sentence-transformers` | `all-MiniLM-L6-v2` — 384-dim article embeddings |
| `pandas` | MINDsmall data loading and preprocessing |
| `numpy` | Numerical computation |
| `tqdm` | Progress tracking |

---

## 🔮 Future Work

- **Offline evaluation with IPS/DR** — current offline evaluation on training data may overfit
- **Attention-based state representation** — transformer over click history instead of concat
- **Actor-Critic architecture** — SAC or A2C better suited for large action spaces
- **Multi-agent RL** — model content creators and recommender as competing agents
- **Decision Transformers** — sequence modeling approach for offline RL on news data
- **A/B testing** — live evaluation on real platform to measure actual engagement lift

---

## 📄 References

1. Wu et al. (2020). *MIND: A Large-scale Dataset for News Recommendation.* arXiv:2003.01243
2. Reimers & Gurevych (2019). *Sentence-BERT.* arXiv:1908.10084
3. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature 518
4. Weng et al. (2021). *Tianshou: A Highly Modularized Deep Reinforcement Learning Library.* JMLR
5. Pathak et al. (2017). *Curiosity-driven Exploration by Self-supervised Prediction.* ICML

---

## 🤝 Contributions

Yogeshvar Reddy Kallam and Ajay Krishna Devulapally contributed equally — problem formulation, data pipeline, all RL agent implementations (MC/Q-Learning/SARSA/DQN/DQN+ICM), state representation design, hyperparameter tuning, experiments, and report writing.

---

*See also: [ist597-deep-rl-homeworks](https://github.com/yogeshvarreddykallam/ist597-deep-rl-homeworks) for all homework implementations.*
