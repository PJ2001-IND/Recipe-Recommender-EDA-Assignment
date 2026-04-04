# 🍽️ Recipe Recommender — EDA & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![NLP](https://img.shields.io/badge/NLP-Feature%20Extraction-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> An end-to-end **Exploratory Data Analysis and Feature Engineering pipeline** for a Recipe Recommender System — covering data cleaning, interaction analysis, nutritional feature extraction, and NLP-based ingredient/tag processing to build a foundation for personalised recipe recommendations.

---

## 📌 Problem Statement

Recommender systems for food and recipes need to understand both **what users like** (interaction data) and **what recipes contain** (content data — ingredients, nutrition, tags, cooking time). Raw recipe datasets are messy, text-heavy, and require careful feature engineering before any recommendation model can be trained.

This project builds the complete data preparation and feature extraction pipeline for a recipe recommender system — the critical groundwork that determines recommendation quality.

---

## 🎯 Objective

- Perform thorough EDA on recipe and user interaction data to uncover patterns and data quality issues
- Clean and preprocess both the recipes and interactions datasets
- Extract meaningful numerical and categorical features from raw recipe attributes
- Apply NLP techniques to process ingredients, tags, and recipe descriptions
- Build a feature-rich dataset ready for training a recommendation model

---

## 📂 Dataset

| File | Description |
|---|---|
| `RAW_recipes_cleaned.csv` | Cleaned recipe dataset — recipe ID, name, ingredients, nutrition, tags, steps, cooking time |
| `RAW_interactions_cleaned.csv` | Cleaned user interaction dataset — user ID, recipe ID, rating, review, date |

### Key Features in Recipes Dataset:
- Recipe metadata: name, cooking time, number of steps, number of ingredients
- Nutritional information: calories, fat, sugar, sodium, protein, saturated fat, carbohydrates
- Text features: ingredient lists, tags, step-by-step instructions, description

### Key Features in Interactions Dataset:
- User-recipe ratings (1–5 scale)
- Review text
- Interaction timestamps

---

## 🗂️ Pipeline Overview

This project follows a **structured 3-notebook pipeline**:

```
RAW_recipes_cleaned.csv + RAW_interactions_cleaned.csv
                │
                ▼
┌─────────────────────────────────────────┐
│  Notebook 01: Feature Extraction Pt. 1  │
│  ├── Data loading & initial inspection  │
│  ├── Missing value analysis             │
│  ├── Numerical feature extraction       │
│  │   (cooking time, n_steps, n_ingreds) │
│  └── Nutrition vector parsing           │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Notebook 02: EDA — Complete Solution   │
│  ├── Rating distribution analysis       │
│  ├── Most popular recipes & cuisines    │
│  ├── Cooking time vs rating correlation │
│  ├── User activity patterns             │
│  ├── Ingredient frequency analysis      │
│  └── Nutritional pattern visualisation  │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Notebook 03: Feature Extraction Pt. 2  │
│  ├── NLP processing of ingredients      │
│  ├── Tag encoding & vectorisation       │
│  ├── TF-IDF on recipe descriptions      │
│  ├── User-recipe interaction matrix     │
│  └── Final feature matrix assembly      │
└─────────────────────────────────────────┘
                │
                ▼
     Feature-Rich Dataset
  (Ready for Recommendation Model)
```

---

## 🔬 Key Analyses Performed

**Exploratory Data Analysis**
- Distribution of user ratings — is the data skewed towards positive reviews?
- Most frequently reviewed and highest-rated recipes
- Correlation between cooking time, number of steps, and user ratings
- User engagement patterns — power users vs casual raters

**Feature Engineering**
- Parsing the nutrition column (stored as a string list) into 7 separate numerical features
- Extracting and normalising cooking time, number of steps, and ingredient count
- TF-IDF vectorisation of ingredient lists for content-based similarity
- Tag frequency encoding for cuisine type, dietary tags, and meal category
- Constructing a sparse user-recipe interaction matrix for collaborative filtering

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical feature processing |
| Matplotlib / Seaborn | EDA visualisations |
| Scikit-learn | TF-IDF vectorisation, feature scaling |
| NLTK / re | Text preprocessing and NLP |
| Jupyter Notebook | Interactive multi-stage pipeline |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

### Run the Pipeline (in order)

```bash
# Clone the repository
git clone https://github.com/PJ2001-IND/Recipe-Recommender-EDA-Assignment.git

# Navigate to the project directory
cd Recipe-Recommender-EDA-Assignment

# Step 1 — Feature Extraction Part 1
jupyter notebook 01_FeatureExtractionPart01-TemplateV4.0.ipynb

# Step 2 — EDA
jupyter notebook 02_EDA-CompleteSolution.ipynb

# Step 3 — Feature Extraction Part 2
jupyter notebook 03_FeatureExtractionPart02-CompleteSolution.ipynb
```

> ⚠️ Run the notebooks **in order** — each notebook builds on the outputs of the previous one.

---

## 📁 Project Structure

```
📦 Recipe-Recommender-EDA-Assignment
 ┣ 📓 01_FeatureExtractionPart01-TemplateV4.0.ipynb         # Initial feature extraction
 ┣ 📓 02_EDA-CompleteSolution.ipynb                         # Full exploratory analysis
 ┣ 📓 03_FeatureExtractionPart02-CompleteSolution.ipynb     # NLP & advanced feature engineering
 ┣ 📄 RAW_recipes_cleaned.csv                               # Cleaned recipe dataset
 ┣ 📄 RAW_interactions_cleaned.csv                          # Cleaned user interactions dataset
 ┣ 📓 Recipe_Recommender_assignment.ipynb                   # Combined assignment notebook
 ┗ 📄 README.md                                             # Project documentation
```

---

## 💡 Key Insights

- Recipe ratings are heavily skewed towards 4–5 stars, reflecting a positivity bias in user reviews — this needs to be accounted for during model training
- Shorter recipes (fewer steps, under 30 minutes) consistently receive higher average ratings, suggesting convenience is a strong satisfaction driver
- Ingredient overlap is a powerful signal — recipes sharing rare or specific ingredients tend to attract similar user cohorts
- NLP-based tag encoding captures cuisine and dietary preferences far better than simple categorical encoding

---

## 🔭 Future Scope

- Build a **collaborative filtering model** (matrix factorisation / SVD) using the user-recipe interaction matrix
- Implement **content-based filtering** using TF-IDF ingredient similarity for cold-start users
- Develop a **hybrid recommender** combining collaborative and content-based signals
- Deploy as an interactive **Streamlit app** where users can input preferences and get recipe recommendations in real time

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you found this project useful, consider giving it a star!
