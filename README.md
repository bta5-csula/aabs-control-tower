# AABS Control Tower v8.3
## Enterprise Decision Intelligence Platform

> [!TIP]
> **Python Version Note**: This project works best with **Python 3.10 - 3.12**. Using Python 3.14+ may cause slow installations as libraries must be compiled from source.

---

## 🌟 Executive Summary (For Non-Technical Users)

### **What is the AABS Control Tower?**
Imagine a "Flight Control Center" but for a global business. Large companies often struggle to see what is *actually* happening across their thousands of shipments, warehouses, and sales offices. The AABS Control Tower acts as the **central nervous system**, pulling in data from all these different places into one single, high-tech dashboard.

### **What problem does it solve?**
Traditional business reports tell you what happened *last month*. The AABS Control Tower tells you what is happening **right now** and what might go wrong **tomorrow**.
*   **Late Deliveries**: It detects "at-risk" orders before they are even shipped.
*   **Market Disruptions**: It tracks real-time traffic jams, satellite activity at warehouses, and market price changes.
*   **Revenue Protection**: It alerts managers when sales are falling behind plan, so they can act before the quarter ends.

### **Key Benefits:**
1.  **Faster Decisions**: Move from "guessing" to "knowing" in seconds.
2.  **Risk Mitigation**: Identify supply chain bottlenecks before they cost money.
3.  **Proactive Strategy**: Use AI to forecast demand and plan your inventory accurately.

---

## 🚀 Quick Start (3 Steps)

### Option A: One-Click Start (Recommended)

**Windows:**
Just double-click or run:
```cmd
start.bat
```

**Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

### 🔁 Stopping & Restarting
*   **To Stop**: Press **`Ctrl + C`** in your terminal window.
*   **To Restart**: Simply run **`start.bat`** again. The system will skip the installation steps and launch instantly.

---

## 📁 Technical Architecture (Modular)

The project is organized for professional maintenance and scalability:

*   **`app.py`**: The main application "brain" and user interface.
*   **`core/config.py`**: Centralized settings, branding, and file paths.
*   **`assets/style.css`**: Professional "Premium Dark Mode" design tokens and animations.
*   **`ml/`**: Pre-trained Random Forest models for Risk and Demand forecasting.
*   **`memory/`**: SQLite database that tracks AI recommendations and user actions over time.
*   **`uploads/`**: Secure location for your business data files.

---

## 📊 What It Does

| Tab | Description |
|-----|-------------|
| 📦 **Logistics** | VA05 order risk analysis, late delivery prediction |
| 💰 **Finance** | Plan vs Actual variance, customer risk scoring |
| 📈 **Forecasting** | Ensemble ML predictions with 95% confidence bands |
| 🌐 **External Signals** | Real-time traffic, satellite, market intelligence |
| 🚨 **Alert Center** | Unified alerts, executive briefing, report download |
| 🎮 **Scenarios** | What-if stress testing for market disruptions |
| 🔌 **Data Sources** | API status and database health monitoring |
| 🎯 **Strategy** | ERPSIM competitive analysis and market share |

---

## ✨ Advanced Features

### **Real-Time Intelligence**
*   **Traffic Signals**: Monitors 6 major logistics corridors.
*   **Satellite Activity**: Tracks utilization at 5 key distribution centers.
*   **Market Stress**: Aggregates steel, fuel, and consumer confidence indexes.

### **Ensemble Machine Learning**
*   **Risk Scoring**: Random Forest Classifier with 99% recall on historical bottlenecks.
*   **Demand Forecasting**: Regressor model with R² = 0.981 for high-accuracy inventory planning.

---

## 🔌 Going Live with Real APIs
The platform uses high-fidelity mock data by default. To enable live API feeds:

```bash
# Set API keys in your environment
export GOOGLE_MAPS_API_KEY="your-key"
export ALPHA_VANTAGE_API_KEY="your-key"
export OPENWEATHER_API_KEY="your-key"

# Disable mock mode
export AABS_MOCK_MODE="false"
```

---

## 🎯 For Recruiters
> "I built an enterprise decision intelligence platform that aggregates internal ERP data with real-time external signals. Features include **Random Forest ML models** for risk scoring, a **SQLite-backed memory system** for tracking decision outcomes, and a **high-performance modular architecture**. Built with real enterprise data: 171,000 transaction records over 13 years."

---

**Built by AABS | Enterprise Decision Intelligence**
*"The nervous system of the modern economy"*
