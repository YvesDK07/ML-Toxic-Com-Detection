# Toxic Comment Detection 🛡️

> **Advanced ML ensemble system for toxic comment detection with explainable AI**  
> EFREI Paris | TI508M/TI608M Introduction to Machine Learning

**Authors:** Yves de KERROS • Warren Assepo

---

## 🎯 Key Results

| Metric | Value | Status |
|--------|-------|--------|
| **Best Model (Neural Network)** | 98.24% | ⭐ Excellent |
| **Ensemble (XGB+SVC+MLP)** | 96.93% | ✅ Production-ready |
| **Target Achievement** | 85% → 98.24% | 🎯 +13.24% |
| **Training Time** | ~10 minutes | ⚡ Fast |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/toxic-comment-detection.git
cd toxic-comment-detection

# Run main script
python toxic-com-detection.py
```

---

## 📊 What We Built

**Intelligent toxicity detection system with:**

- 🎓 **6 ML models** trained and compared (XGBoost, Random Forest, Linear SVC, Decision Tree, Neural Network, Ensemble)
- 🔍 **1-10 toxicity scoring** with severity levels (non-toxic → extremely toxic)
- 💡 **Explainable AI** showing which words trigger toxicity detection
- ⚖️ **SMOTE balancing** solving the 90-10 class imbalance without double-weighting
- 📈 **160K+ comments** from Jigsaw Kaggle + YouToxic datasets

---

## 🏆 Key Innovation

**Solved the SMOTE + XGBoost double-weighting problem:**

```python
# ❌ Common mistake: 70% accuracy
smote = SMOTE()
model = XGBClassifier(scale_pos_weight=8.9)  # Double-weighting!

# ✅ Our solution: 88.67% accuracy
smote = SMOTE()
model = XGBClassifier(scale_pos_weight=1)    # Balanced correctly
```

This discovery improved XGBoost accuracy from ~70% to **88.67%** 🎯

---

## 🔬 Technical Stack

**Preprocessing:** TF-IDF (5000 features, bigrams, min_df=2, max_df=0.95)  
**Balancing:** SMOTE (50-50 distribution)  
**Models:** XGBoost, Random Forest, LinearSVC, Decision Tree, MLP, Ensemble  
**Best Architecture:** Neural Network (256→128→64, ReLU, 50 epochs)

---

## 📈 Model Comparison

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Neural Network | **98.24%** | ~8 min |
| Ensemble (XGB+SVC+MLP) | **96.93%** | ~10 min |
| Linear SVC | **92.18%** | ~3 min |
| XGBoost | **88.67%** | ~5 min |
| Random Forest | 82.24% | ~6 min |
| Decision Tree | 70.89% | ~2 min |

**All models trained on:** 288,658 balanced samples (80-20 split)

---

## 🎨 Example Predictions

```python
Comment: "Thank you for the amazing content!"
→ Score: 2.5/10 | Severity: Non-toxic 🟢

Comment: "This is absolutely disgusting"  
→ Score: 6.7/10 | Severity: Highly toxic 🔴

Comment: "You are a f***ing idiot"
→ Score: 10.0/10 | Severity: Extremely toxic ⚫
→ Toxic words: ['fuck', 'idiot'] (with importance weights)
```

---

## 📚 Documentation

- **Code:** Fully commented `toxic-com-detection.py`
- **Dataset:** [Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) + YouToxic English 1000

---

## 🎓 Academic Context

**Course:** TI508M/TI608M - Introduction to Machine Learning  
**Institution:** EFREI Paris  
**Evaluation:** Project achieved all objectives (85%+ accuracy target exceeded)  
**Date:** December 2024

---

## 🙏 Acknowledgments

- MAHE ML course instructors
- Kaggle Jigsaw Toxic Comment dataset
- YouToxic dataset contributors
- XGBoost, scikit-learn, and pandas communities

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Built with ❤️ by Yves de KERROS & Warren Assepo | EFREI Paris 2024

</div>
