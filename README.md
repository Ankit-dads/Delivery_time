# 🍔 Food Delivery Time Estimation (ETA Prediction)

A end-to-end Machine Learning regression project that predicts food delivery time (in minutes) based on real-world factors like distance, traffic, weather, and delivery agent details.

---

## 📌 Problem Statement

Food delivery platforms like Swiggy and Zomato need accurate ETA predictions to improve customer experience. This project builds a regression model that estimates delivery time using order-level and environmental features.

**Target Variable:** `Time_taken(min)` — the actual delivery time in minutes.

---

## 📂 Dataset

- **Source:** [Food Delivery Time Dataset — Google Drive](https://drive.google.com/drive/folders/1XE_pqKqu0GlkRP5yLXlHrCPmrnbfgOZY)
- **Size:** 45,593 rows × 20 columns
- **Type:** Supervised Regression

| Feature | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery agent |
| `Delivery_person_Ratings` | Agent rating (1–5 scale) |
| `Restaurant_latitude/longitude` | Restaurant GPS coordinates |
| `Delivery_location_latitude/longitude` | Customer GPS coordinates |
| `Order_Date` | Date of the order |
| `Time_Orderd` | Time when order was placed |
| `Time_Order_picked` | Time when agent picked up the order |
| `Weatherconditions` | Sunny, Stormy, Cloudy, Fog, etc. |
| `Road_traffic_density` | Low / Medium / High / Jam |
| `Vehicle_condition` | Condition score of the vehicle |
| `Type_of_order` | Snack / Meal / Drinks / Buffet |
| `Type_of_vehicle` | Motorcycle / Scooter / etc. |
| `multiple_deliveries` | Number of simultaneous deliveries |
| `Festival` | Whether it's a festival day |
| `City` | Urban / Metropolitian / Semi-Urban |

---

## 🔧 Project Workflow

```
Data Loading → Data Cleaning → Feature Engineering
→ EDA → Encoding → Model Training → Evaluation → Hyperparameter Tuning
```

### 1. Data Cleaning
- Extracted numeric values from messy string formats (`"(min) 24"` → `24`, `"conditions Sunny"` → `"Sunny"`)
- Stripped trailing whitespace from all categorical columns
- Removed literal `'NaN'` string rows
- Converted `Age`, `Ratings`, `multiple_deliveries` from object to numeric using `pd.to_numeric(errors='coerce')`
- Dropped remaining null rows after all conversions

### 2. Feature Engineering
Four new features were derived to improve model accuracy:

| New Feature | How it's derived | Why it matters |
|---|---|---|
| `distance_km` | Haversine formula on restaurant & delivery GPS coords | Main driver of delivery time |
| `prep_time_min` | `Time_Order_picked` − `Time_Orderd` in minutes | Captures kitchen delay |
| `order_hour` | Hour extracted from `Time_Orderd` | Rush hour effect |
| `day_of_week` | Day extracted from `Order_Date` | Weekend vs weekday patterns |
| `is_weekend` | Binary flag from `day_of_week` | Higher demand on weekends |

### 3. Exploratory Data Analysis (EDA)
- Distribution of delivery time (target variable)
- Distance vs delivery time (scatter plot)
- Traffic density vs delivery time (box plot)
- Weather conditions vs delivery time (box plot)
- City type vs delivery time (box plot)
- Correlation heatmap of all numeric features

### 4. Model Building & Comparison

Four regression models were trained and compared:

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 5.31 | 6.60 | 0.5123 |
| Decision Tree | 3.97 | 5.20 | 0.6979 |
| Random Forest | 3.07 | 3.81 | 0.8375 |
| Gradient Boosting | 3.40 | 4.25 | 0.7982 |

> Results filled after final model run.

### 5. Hyperparameter Tuning
- `RandomizedSearchCV` on Random Forest with `n_iter=20`, `cv=5`
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

### 6. Cross Validation
- 5-fold cross validation on best model
- Reported mean R² ± standard deviation

---

## 📊 Key Insights

- **Distance** is the strongest predictor of delivery time (confirmed via feature importance)
- **Traffic density** significantly increases delivery time, especially at `Jam` level
- **Festival days** add measurable delay compared to normal days
- **Prep time** (pickup delay) contributes meaningfully to total ETA
- **Weather** conditions like Stormy and Sandstorm correlate with longer deliveries

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib / Seaborn | EDA visualizations |
| Scikit-learn | ML models, preprocessing, evaluation |
| math (Haversine) | Distance calculation from GPS |

---

## 📁 File Structure

```
food-delivery-eta/
│
├── food_eta_prediction.ipynb   ← Main notebook (full pipeline)
├── train.csv                   ← Dataset
└── README.md                   ← Project documentation
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Ankit-dads/Delivery_time.git
cd food-delivery-eta

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Open the notebook
jupyter notebook food_eta_prediction.ipynb
```

---

## 📈 Evaluation Metric

Primary metric: **R² Score** (coefficient of determination)  
Supporting metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)

R² measures what percentage of variance in delivery time the model explains. A score above **0.80** is the target for this dataset.

---

## 👤 Author

**Ankit Kashyap**  
Aspiring Data Scientist | Python · SQL · Scikit-learn · Power BI  
🔗 [GitHub](https://github.com/Ankit-dads) · [LinkedIn](https://www.linkedin.com/in/ankit-kashyapp/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
