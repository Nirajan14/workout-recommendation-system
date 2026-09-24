# Workout Recommendation System

A Flask web app that recommends a real workout program and predicts calories burned per session, based on the user's goal, experience level, equipment access, and body stats.

## How it works

1. **User input** — age, gender, height, weight, workout days/week, session duration, fitness goal, experience level, and equipment access (`templates/input.html`).
2. **Calorie prediction** — a scikit-learn regression model (`model/calorie_model.pkl`) trained on `data/gym_members_exercise_tracking.csv` predicts expected calories burned per session.
3. **Program recommendation** — a TF-IDF + cosine-similarity matcher (`model/program_vectorizer.pkl`, `model/program_tfidf_matrix.pkl`) ranks real workout programs from `data/program_summary_clean.csv` against the user's goal/level/equipment, then merges the top matches into one composite weekly plan with real exercises, sets, and reps.
4. **Result page** — shows predicted calories, BMI, and the generated plan (`templates/result.html`).

## Project structure

```
app.py                          Flask app (routes, prediction, recommendation logic)
requirements.txt                Python dependencies
data/
  gym_members_exercise_tracking.csv   Raw data used to train the calorie model
  programs_detailed.csv               Raw exercise-level program data (1 row per exercise/day)
                                       NOT included in this repo (~282MB, exceeds GitHub's limit)
                                       Download: https://drive.google.com/file/d/1wUfPL55LlocN8O5ivMMXf_9tLngk8Fi-/view?usp=drive_link
                                       Place it in data/ before running notebooks/train.ipynb.
  program_summary_clean.csv           One row per program, built from programs_detailed.csv
model/
  calorie_model.pkl                   Trained calorie regression model (not committed — see Setup)
  calorie_feature_order.pkl           Feature order expected by the calorie model (not committed)
  calorie_metrics.txt                 Evaluation metrics (MAE, R²) for the calorie model
  program_vectorizer.pkl              TF-IDF vectorizer fit on program_summary_clean.csv (not committed)
  program_tfidf_matrix.pkl            TF-IDF matrix aligned with program_summary_clean.csv (not committed)
  program_catalog.pkl                 Pickled copy of the cleaned program catalog (not committed)
notebooks/
  train.ipynb                         Full training pipeline: data cleaning, EDA, model
                                       training, evaluation graphs, and sample predictions
templates/                            HTML pages (Jinja2)
static/                               CSS
REPORT.docx                           Project write-up
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

**Note:** `model/*.pkl` files are not committed to this repo (regenerated artifacts). Before running `app.py` for the first time, run `notebooks/train.ipynb` end-to-end to generate them — you'll also need `data/programs_detailed.csv` (see above) for that.

## Retraining

Open `notebooks/train.ipynb` and run all cells. It will:

- Aggregate `programs_detailed.csv` (exercise-level) into `data/program_summary_clean.csv` (one row per program)
- Retrain the TF-IDF program recommender and the calorie regression model
- Regenerate all files in `model/` and `data/program_summary_clean.csv`
- Show EDA plots, evaluation graphs (actual vs. predicted, residuals, feature importance), and a sample prediction

**Important:** `app.py` reads `data/program_summary_clean.csv` — a per-program summary — not the raw `programs_detailed.csv`. If you delete or regenerate data, always re-run the notebook so this file and the model artifacts stay in sync (they must share the same row order).

## Notes

- The calorie model only uses: Age, Gender, Weight, Height, Session Duration, Workout Frequency, Experience Level, BMI.
- The program recommender works purely on text similarity (TF-IDF) between the user's goal/level/equipment and each program's description — there's no manual rule table beyond keyword weighting in `app.py`.
- --
