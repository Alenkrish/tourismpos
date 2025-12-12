# 🌍 Sri Lanka Tourism Prediction & Recommendation App

This project is a Streamlit-based web application that predicts monthly tourist arrivals in Sri Lanka and recommends the best tourist attractions based on the selected month.  
It uses machine learning models, cleaned datasets, and hidden-gem scoring to create an intelligent eco–cultural tourism suggestion system.

---

## 🧩 Features

### ✅ 1. Tourist Arrival Prediction  
- Predicts national monthly tourist arrivals using a Random Forest Regressor  
- Automatically falls back to historical monthly averages if model is missing  
- Displays trend charts for better seasonal understanding  

### ✅ 2. Attraction Recommendation System  
- Suggests the best attractions for a selected month  
- Uses ML-based popularity scoring  
- Hidden-gem scoring based on uniqueness, reviews, and eco/cultural attributes  
- Optional filters: category, month, and hidden-preference  
- Shows attractions on a map  

### ✅ 3. Clean, Robust Architecture  
- Lazy-loaded models (prevents blank-screen crashes)  
- Safe CSV loading with warnings instead of errors  
- Detailed suggestion breakdown and downloadable results  

---

## 📁 Project Structure

