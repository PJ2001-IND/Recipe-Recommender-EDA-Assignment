# 🍽️ Recipe Recommender System — EDA & Feature Engineering (PySpark)

![PySpark](https://img.shields.io/badge/PySpark-Big%20Data-E25A1C?style=flat-square&logo=apache-spark)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![AWS EMR](https://img.shields.io/badge/AWS-EMR%20%7C%20S3-FF9900?style=flat-square&logo=amazon-aws)

> A **large-scale recipe recommendation system pipeline** built entirely on **Apache PySpark** — processing **231,637 recipes** and **1,132,367 user interactions** from food.com. The project covers end-to-end feature extraction, EDA, nutrition standardisation, time-based feature engineering, tag-level analysis with OHE, and user behaviour feature construction — all designed to produce a model-ready dataset for downstream recommendation modelling.

---

## 📌 Problem Statement

Food recommendation platforms deal with massive datasets where traditional pandas-based processing fails to scale. This project builds a production-grade PySpark pipeline on food.com data to:

- **Extract and engineer structured features** from raw recipe and interaction datasets
- **Understand what drives high ratings** — by analysing prep time, ingredients, nutrition, recipe age, and tags
- **Build user-level behavioural features** — capturing each user's rating patterns, preferred prep times, and nutrition preferences
- **Produce a model-ready parquet dataset** that can be directly fed into a recommendation model (collaborative filtering or content-based)

---

## 🎯 Objectives

- Read and validate **231,637 recipes** and **1,132,367 user interactions** from S3/local storage using PySpark
- Extract 7 individual nutrition values from a packed string column and standardise them to **per-100-calorie** units
- Convert the `tags` string column to `ArrayType(StringType())` for tag-level analysis
- Compute **time-based features**: days, months, and years since recipe submission on each review date
- Perform **EDA with quantile bucketing** on prep time, steps, ingredients, and all nutrition columns
- Engineer **user-level aggregate features** — average ratings, prep time preferences, and nutrition affinity
- Derive **5-star-rating-specific user features** to capture high-preference behavioural patterns
- Apply **tag-level OHE** — one-hot encode top frequent tags, top rated tags, and bottom rated tags as binary features

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Source | food.com (formerly Genius Kitchen) — publicly available recipe dataset |
| Storage | AWS S3 / local filesystem — read via PySpark |
| Infrastructure | Apache Spark on AWS EMR — `pysparkkernel` |

### 1. RAW_recipes_cleaned.csv — Recipe Data

| Property | Detail |
|---|---|
| File | `RAW_recipes_cleaned.csv` |
| Records | **231,637 recipes** |
| Columns | 12 (original) → 25 (after feature extraction) |
| Key Fields | `id`, `name`, `minutes`, `tags`, `nutrition`, `n_steps`, `n_ingredients`, `submitted` |

### 2. RAW_interactions_cleaned.csv — User Ratings Data

| Property | Detail |
|---|---|
| File | `RAW_interactions_cleaned.csv` |
| Records | **1,132,367 user interactions** |
| Columns | 5 (`user_id`, `recipe_id`, `date` → `review_date`, `rating`, `review`) |
| Merged Dataset | **1,132,367 rows × 30 columns** (inner join on `recipe_id = id`) |

### Key Schema After Feature Engineering

| Column | Type | Description |
|---|---|---|
| `calories` | Float | Calories per serving |
| `total_fat_PDV` … `carbohydrates_PDV` | Float | 6 raw nutrition values (PDV) |
| `total_fat_per_100_cal` … `carbohydrates_per_100_cal` | Float | 6 standardised nutrition values per 100 cal |
| `tags` | Array\<String\> | Recipe tags parsed from string to array |
| `days_since_submission_on_review_date` | Integer | Days between submission and review |
| `months_since_submission_on_review_date` | Float | Months between submission and review |
| `years_since_submission_on_review_date` | Float | Years between submission and review |

---

## 🔬 Methodology

```
RAW_recipes_cleaned.csv (231,637 recipes, 12 cols)     RAW_interactions_cleaned.csv (1,132,367 rows, 5 cols)
   │                                                          │
   ▼                                                          │
Notebook 01 — Feature Extraction Part 1                       │
   │   ├── Task 1: Read recipes → raw_recipes_df              │
   │   │     Assert: 231,637 rows, 12 cols, correct dtypes    │
   │   ├── Task 2: Extract nutrition string → 7 float cols    │
   │   │     Strip brackets → split on comma → cast FloatType │
   │   │     [calories, total_fat_PDV, sugar_PDV, sodium_PDV, │
   │   │      protein_PDV, saturated_fat_PDV, carbs_PDV]      │
   │   ├── Task 3: Standardise nutrition → per 100 cal        │
   │   │     col_per_100_cal = col_PDV * 100 / calories       │
   │   ├── Task 4: Convert tags string → ArrayType(String)    │
   │   │     Remove [ ] ' → split on comma                    │
   │   ├── Task 5: Join recipes + ratings (inner on recipe_id)│◄─┘
   │   │     Assert: 1,132,367 rows × 30 cols                 │
   │   ├── Task 6: Create time-based features                 │
   │   │     Cast submitted & review_date → DateType          │
   │   │     datediff → days_since_submission                  │
   │   │     months_between → months_since_submission          │
   │   │     months / 12 → years_since_submission             │
   │   └── Save → interaction_level_df.parquet (33 cols)      │
   │                                                          │
   ▼                                                          │
Notebook 02 — EDA                                             │
   │   ├── Read interaction_level_df.parquet                  │
   │   │     Assert: 1,132,367 rows × 33 cols                 │
   │   ├── years_since_submission: filter out negatives       │
   │   │     Buckets: [0,1), [1,3), [3,6), [6,∞)             │
   │   ├── minutes (prep time): cap at 930 min, remove 0s     │
   │   │     Buckets: [0,5), [5,15), [15,30), [30,60),        │
   │   │              [60,300), [300,900), [900,∞)            │
   │   ├── n_steps: remove 0-step recipes                     │
   │   │     Buckets: [0,2), [2,6), [6,8), [8,12),            │
   │   │              [12,29), [29,∞)                         │
   │   ├── n_ingredients:                                     │
   │   │     Buckets: [0,6), [6,9), [9,11), [11,∞)           │
   │   ├── Nutrition columns (13 total): quantile bucketing   │
   │   │     Splits at 0.25, 0.50, 0.75, 0.95 quantiles      │
   │   └── Save → interaction_level_df_postEDA.parquet        │
   │                                                          │
   ▼
Notebook 03 — Feature Extraction Part 2
   │   ├── Read interaction_level_df_postEDA.parquet
   │   ├── User-level aggregate features (Window by user_id)
   │   │     user_avg_rating, user_n_ratings
   │   │     user_avg_years_betwn_review_and_submission
   │   │     user_avg_prep_time_recipes_reviewed
   │   │     user_avg_{nutrition_col}_recipes_reviewed (×7)
   │   ├── 5-star-specific user features
   │   │     user_n_5_ratings
   │   │     user_avg_years_betwn_review_and_submission_5_ratings
   │   │     user_avg_prep_time_recipes_reviewed_5_ratings
   │   │     user_avg_n_steps_recipes_reviewed_5_ratings
   │   │     user_avg_{nutrition_col}_recipes_reviewed_5_ratings (×7)
   │   ├── Tag-level EDA + OHE
   │   │     Explode tags → groupBy tag → avg/count ratings
   │   │     Top frequent tags (>75th pctile interactions, >0.16)
   │   │     Top rated tags (>100 reviews, avg_rating > 4.53)
   │   │     Bottom rated tags (avg_rating < 4.00)
   │   │     → add_OHE_columns() → has_tag_{name} binary cols
   │   └── Save → interaction_level_df_ModelReady.parquet
```

---

## 📊 Dataset Facts (Confirmed by Assert Statements)

| Checkpoint | Value |
|---|---|
| Raw recipes | 231,637 rows, 12 columns |
| Raw interactions | 1,132,367 rows, 5 columns |
| After join (Task 5) | 1,132,367 rows × **30 columns** |
| After time features (Task 6) | 1,132,367 rows × **33 columns** |
| Post-EDA parquet | 1,132,367 rows (after filtering negatives, 0-min, 0-step) |
| User: `user_id 601529` avg rating | 4.22 (27 reviews) |
| User: `user_id 233044` n_5_ratings | 7 |

> 📝 *Refer to the 4 notebooks for full task solutions, assert validations, bucketwise plots (`%matplot plt`), and the final model-ready schema.*

---

## 💡 Key Insights

- **Recipes more than 6 years old** receive lower ratings — review age is a meaningful signal
- **Short prep time is preferred** — recipes under 30 minutes consistently receive higher ratings than those over 60 minutes
- **Fewer steps correlates with higher ratings** — recipes with ≤2 steps are rated highest; those with >29 steps rated lowest
- **Nutrition columns are not predictive of ratings** — EDA confirms calories, fat, sugar, sodium, protein, saturated fat, and carbohydrates show no strong relationship with user ratings
- **Tag OHE captures taste preferences** — top 5-percentile frequently-reviewed tags and high-average-rated tags (>4.53) provide discriminative binary features for the recommender
- **Rare tags are not OHE-encoded** — tags appearing in <100 interactions would produce near-zero columns with no modelling value
- **User behaviour features are powerful** — capturing what a user *typically reviews* (prep time, nutrition, n_steps) provides strong personalisation signals

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Apache PySpark | Distributed data processing — all transformations at scale |
| PySpark MLlib (`Bucketizer`) | Quantile-based bucketing of numerical features |
| PySpark Window Functions | User-level aggregate feature computation |
| PySpark SQL Functions | `F.regexp_replace`, `F.split`, `F.datediff`, `F.months_between`, `F.array_contains`, `F.explode` |
| Pandas | Local summary statistics and bucket-level EDA tables |
| NumPy | Numerical operations for quantile summaries |
| Matplotlib | Bucketwise rating visualisation plots |
| AWS EMR / S3 | Cloud infrastructure — Spark cluster and data storage |
| Jupyter Notebook | Interactive development environment (pysparkkernel) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pyspark pandas numpy matplotlib pyarrow
```

### Run the Notebooks (in order)

```bash
# Clone the repository
git clone https://github.com/PJ2001-IND/Recipe-Recommender-EDA-Assignment.git

# Navigate to the project directory
cd Recipe-Recommender-EDA-Assignment

# Run notebooks in sequence:
# 1. Feature Extraction Part 1
jupyter notebook 01_FeatureExtractionPart01-TemplateV4_0.ipynb

# 2. EDA
jupyter notebook 02_EDA-CompleteSolution.ipynb

# 3. Feature Extraction Part 2
jupyter notebook 03_FeatureExtractionPart02-CompleteSolution.ipynb

# 4. Combined Recommender Pipeline
jupyter notebook Recipe_Recommender_assignment.ipynb
```

> ⚠️ **Note:** All notebooks use `pysparkkernel` — a running Spark session is required. Data is read from S3 paths (`s3a://`) in the original notebooks. Update the paths to point to local copies of `RAW_recipes_cleaned.csv` and `RAW_interactions_cleaned.csv` for local execution. The two CSV files are stored via **Git LFS** — run `git lfs pull` after cloning to download the full datasets.

---

## 📁 Project Structure

```
📦 Recipe-Recommender-EDA-Assignment
 ┣ 📓 01_FeatureExtractionPart01-TemplateV4_0.ipynb   # Tasks 1–6: Read, extract nutrition, tags, join, time features
 ┣ 📓 02_EDA-CompleteSolution.ipynb                   # EDA: bucketing prep time, steps, ingredients, nutrition
 ┣ 📓 03_FeatureExtractionPart02-CompleteSolution.ipynb # User features, 5-star features, tag OHE
 ┣ 📓 Recipe_Recommender_assignment.ipynb             # Full combined pipeline (all 3 notebooks consolidated)
 ┣ 📄 RAW_recipes_cleaned.csv                         # Recipe data (231,637 records, 12 cols) [Git LFS]
 ┣ 📄 RAW_interactions_cleaned.csv                    # User interactions (1,132,367 records, 5 cols) [Git LFS]
 ┣ 📄 requirements.txt                                # Python dependencies
 ┗ 📄 README.md                                       # Project documentation
```

---

## 🔭 Future Scope

- Train a **collaborative filtering model** (ALS — Alternating Least Squares) using the model-ready parquet as input for personalised recipe recommendations
- Build a **content-based recommender** using the tag OHE features and nutrition standardised columns
- Implement a **hybrid recommender** combining collaborative and content-based signals
- Deploy as a **real-time recipe recommendation API** using FastAPI with a pre-trained Spark model
- Extend the tag OHE with **TF-IDF weighting** to better represent rare but meaningful tags
- Incorporate **recipe text (reviews and names)** using NLP embeddings for richer content features

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

> ⭐ If you found this project useful, consider giving it a star!
