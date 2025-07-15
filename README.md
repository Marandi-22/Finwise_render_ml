# 🛡️ FinWise — Master Your Money, Outsmart The Scams

> 🔐 A next-gen financial intelligence app that protects users from UPI frauds and impulse spending, while empowering them with interactive financial literacy tools.

![FinWise Banner](https://t3.ftcdn.net/jpg/07/78/11/08/360_F_778110813_nGqTda2YeQ3IE85xss0YzUGWOozNwC3d.jpg)

---

## 🚀 Overview

**FinWise** is your financial co-pilot — combining fraud detection, behavioral insights, and smart educational content into a sleek mobile app experience. Built for the modern Indian user, FinWise uses AI to **analyze transaction intent**, **flag scams in real-time**, and **train users to think long-term about their spending** — all while keeping the UI fun and intuitive.

---

## 🧠 Key Features

### 1️⃣ Scam Detection Engine
- 🔍 **Hybrid AI Model** using ML + LLM
- 🧾 Analyzes UPI transaction metadata:
  - Amount
  - Urgency
  - New merchant?
  - Known scam phrases
  - Budget check
- 📛 Flags high-risk behavior and explains **why it's suspicious** using LLMs.

### 2️⃣ Pre-Spend Reflection System
- 🧘‍♂️ When about to spend, users input their **payment reason**.
- FinWise returns:
  - ⚠️ A warning if it seems impulsive.
  - ✅ Encouragement if aligned with long-term goals.
- Helps build **emotional resilience** and **financial discipline**.

### 3️⃣ Visual Learning Reels (EduVerse)
- 🎮 Choose-your-own-path storytelling.
- Users navigate scenarios like:
  - Investment traps
  - Fake job offers
  - Get-rich-quick schemes
- 🌱 Learn by **making mistakes in simulation**, not real life.

### 4️⃣ Scam Intelligence & Whitelist Database
- 🕵️ Scrapes verified sources (RBI, CERT-IN, NPCI).
- Maintains:
  - ✅ Trusted UPI IDs
  - ✅ Safe domains & QR metadata
  - 🚫 Known scam keywords & URLs

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React Native + Expo |
| **Backend** | Python (FastAPI or Flask) |
| **AI / ML** | Scikit-learn, LLMs via HuggingFace or OpenRouter |
| **Database** | Firebase / Firestore (app), CSV/JSON (intelligence layer) |
| **Visualization** | Streamlit for web demo |
| **Deployment** | Hugging Face Spaces, GitHub, Android/iOS via Expo Go |

---

## 🛠️ ML Pipeline

- **Feature Inputs:**
  - `amount`, `urgency_level`, `is_trusted_upi`, `is_new_merchant`, `match_known_scam_phrase`, `is_budget_exceeded`, `category`, `time_of_day`, `day_of_week`
- **Enrichment Layer:**
  - Converts raw inputs into smart features using context + scraping.
- **Prediction Layer:**
  - Classifies transaction risk (Safe / Warning / Dangerous)
- **Explanation Layer (LLM):**
  - Explains why a transaction might be harmful.
  - Can suggest safer alternatives.

---

## 🧪 Demo & Screenshots

_(Add your screenshots or Loom demo here!)_

---

## 📈 Roadmap

- [x] MVP Scam Detection Engine (90%+ accuracy target)
- [x] Pre-spend Reflection Flow
- [x] Streamlit Dashboard for Model Testing
- [x] Trusted Merchant Scraper + Database
- [ ] UPI QR Scanner with safety status
- [ ] Gamified Learning Reels Module
- [ ] Integration with personal budget tracker
- [ ] Launch on Play Store and App Store

---

## 🧠 Inspiration

> “Scams prey on urgency. FinWise injects clarity into that moment of chaos.”

- Inspired by rising digital fraud cases in India
- Combines cognitive psychology + LLMs + real-world scam patterns
- Built to **empower users**, not just warn them

---

## 🤝 Team & Credits

**Built with obsession by:**
- 🔧 R3wind (Lead Dev, AI Architect, Strategist)

> Special thanks to the mentors, training center faculty, and every scammer who unknowingly helped train our models.

---

## 📂 Folder Structure (Main Repos)

```bash
FinWise/
├── app/                 # React Native mobile app
├── ml_model_integration/
│   ├── unified_engine.py
│   └── scam_classifier.pkl
├── scraper/             # Trusted merchant and keyword scrapers
├── data/
│   ├── trusted_upi_db.json
│   ├── safe_domains.json
│   └── scam_keywords.json
├── streamlit_demo/      # Streamlit app for testing pipeline
├── utils/
└── README.md
