# 🔧 Predictive Maintenance Dashboard

This Streamlit app predicts the likelihood of equipment failure based on real-time sensor inputs. It was built using a trained Random Forest model and is ideal for industrial operations, manufacturing, and facilities management.

---

## 🚀 Demo Features

✅ Live prediction of machine failure  
✅ 5 customizable sensor inputs  
✅ Visual probability gauge + status flag  
✅ Session-based history tracking  
✅ Downloadable prediction logs (CSV)  
✅ In-app explainer for users and stakeholders  
✅ Deployment-ready (Streamlit Cloud or Hugging Face Spaces)

---

## 🧠 How It Works

- **Model**: Random Forest Classifier trained on labeled equipment data  
- **Inputs**: Five numerical sensors (e.g., temperature, vibration, pressure)  
- **Outputs**: Failure likelihood + classification result  
- **Tools Used**:  
  - `scikit-learn`  
  - `joblib`  
  - `Streamlit`  
  - `pandas`, `numpy`

---

## 🖥️ Getting Started (Local)

1. Clone this repo:
```bash
git clone https://github.com/your-username/predictive-maintenance-app.git
cd predictive-maintenance-app
