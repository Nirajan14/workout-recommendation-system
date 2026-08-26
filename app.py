import ast
import os
import re
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'workout-recommendation-secret-key'

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

vectorizer = joblib.load(os.path.join(MODEL_DIR, 'program_vectorizer.pkl'))
program_matrix = joblib.load(os.path.join(MODEL_DIR, 'program_tfidf_matrix.pkl'))
calorie_model = joblib.load(os.path.join(MODEL_DIR, 'calorie_model.pkl'))
calorie_feature_order = joblib.load(os.path.join(MODEL_DIR, 'calorie_feature_order.pkl'))

CATALOG = pd.read_csv(os.path.join(DATA_DIR, 'program_summary_clean.csv')).fillna('')
GOALS = ['Weight Loss', 'Muscle Gain', 'Endurance', 'General Fitness']
EXPERIENCE = ['Beginner', 'Intermediate', 'Advanced']
EQUIPMENT = ['No Equipment', 'Dumbbells Only', 'Full Gym']
DAYS_OPTIONS = [2, 3, 4, 5, 6]
DURATION_OPTIONS = [30, 45, 60, 75, 90]

GOAL_TERMS = {
    'Weight Loss': 'weight loss fat loss conditioning cardio HIIT',
    'Muscle Gain': 'muscle gain hypertrophy bodybuilding strength powerbuilding',
    'Endurance': 'endurance conditioning athletics cardio performance',
    'General Fitness': 'general fitness full body health conditioning bodyweight',
}
EQUIPMENT_TERMS = {
    'No Equipment': 'bodyweight home at home no equipment',
    'Dumbbells Only': 'dumbbell dumbbells garage gym at home',
    'Full Gym': 'full gym barbell cable machine dumbbell',
}
LEVEL_TERMS = {
    'Beginner': 'beginner novice',
    'Intermediate': 'intermediate',
    'Advanced': 'advanced',
}

SET_RE = re.compile(r'(?P<sets>\d{1,2})\s*[x×]\s*(?P<target>\d{1,3}(?:\s*[-–]\s*\d{1,3})?)(?:\s*(?P<unit>reps?|sec(?:onds?)?|min(?:utes?)?|meters?|m|km))?', re.I)
SETS_OF_RE = re.compile(r'(?P<sets>\d{1,2})\s+sets?\s+(?:of\s+)?(?P<target>\d{1,3}(?:\s*[-–]\s*\d{1,3})?)\s*(?P<unit>reps?)?', re.I)
DAY_RE = re.compile(r'^(?:day\s*\d+|\d+\s*[-–:]\s*.+|rest(?:\s*/.*)?)', re.I)
VOLUME_RE = re.compile(r'^(.+?)\s*[-–:]\s*(\d+)\s*$')


def tags(value):
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return [str(value)] if str(value) else []


def detail_lines(description):
    return [x.strip() for x in str(description).replace('\r', '').split('\n') if x.strip()]


def extract_details(description):
    lines = detail_lines(description)
    schedule = []
    prescriptions = []
    volume = []
    in_setup = False
    in_volume = False

    for line in lines:
        low = line.lower()
        if 'weekly setup' in low or 'weekly schedule' in low or 'workout split' in low:
            in_setup = True
            continue
        if 'muscle groups' in low and 'set' in low:
            in_volume = True
            continue

        if in_setup:
            if DAY_RE.match(line):
                schedule.append(line)
            elif low.endswith(':'):
                in_setup = False

        if in_volume:
            m = VOLUME_RE.match(line)
            if m and int(m.group(2)) <= 100:
                volume.append({'muscle_group': m.group(1).strip(), 'sets_per_week': int(m.group(2))})
            elif low.endswith(':'):
                in_volume = False

        for rx in (SET_RE, SETS_OF_RE):
            for m in rx.finditer(line):
                sets = int(m.group('sets'))
                if sets > 20:
                    continue
                before = line[:m.start()].strip(' -–:;,.()')
                if not before or len(before) > 70:
                    continue
                if re.search(r'\b(example|ex\.|progression|week|range)\b', before, re.I):
                    continue
                name = re.sub(r'^(example|ex\.|for example|e\.g\.)\s*[:\-]?\s*', '', before, flags=re.I).strip()
                if not name:
                    continue
                unit = (m.groupdict().get('unit') or 'reps').strip()
                prescriptions.append({'exercise': name, 'display': f"{sets} × {m.group('target')} {unit}"})

    unique = []
    seen = set()
    for item in prescriptions:
        key = (item['exercise'].lower(), item['display'].lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return {'schedule': schedule, 'prescriptions': unique, 'volume': volume}


def program_text(goal, experience, equipment, days, duration):
    return ' '.join([
        GOAL_TERMS[goal], LEVEL_TERMS[experience], EQUIPMENT_TERMS[equipment],
        f'{days} days week {duration} minutes workout'
    ])


def compatible(row, goal, experience, equipment):
    levels = set(tags(row['level']))
    goals = set(tags(row['goal']))
    equip = str(row['equipment']).strip().lower()
    level_ok = bool(levels.intersection({experience, 'Novice' if experience == 'Beginner' else ''}))
    goal_map = {
        'Muscle Gain': {'Bodybuilding', 'Muscle & Sculpting', 'Powerbuilding', 'Powerlifting'},
        'Endurance': {'Athletics'},
        'General Fitness': {'Bodyweight Fitness', 'Athletics'},
        'Weight Loss': set(),
    }
    goal_ok = bool(goals.intersection(goal_map[goal])) if goal_map[goal] else any(
        term in (str(row['title']) + ' ' + str(row['description'])).lower()
        for term in ('weight loss', 'fat loss', 'fat-loss', 'cutting', 'conditioning')
    )
    equipment_map = {
        'No Equipment': ('at home',),
        'Dumbbells Only': ('dumbbell only', 'garage gym', 'at home'),
        'Full Gym': ('full gym', 'garage gym'),
    }
    equip_ok = any(term in equip for term in equipment_map[equipment])
    return level_ok and goal_ok and equip_ok


def rank_programs(goal, experience, equipment, days, duration, top_n=5):
    query = vectorizer.transform([program_text(goal, experience, equipment, days, duration)])
    scores = cosine_similarity(query, program_matrix).ravel()
    ranked = CATALOG.copy()
    ranked['_score'] = scores
    compatible_mask = ranked.apply(lambda r: compatible(r, goal, experience, equipment), axis=1)
    filtered = ranked[compatible_mask]
    if filtered.empty:
        filtered = ranked
    filtered = filtered.sort_values(['_score', 'total_exercises'], ascending=[False, False])
    return filtered.head(top_n)


def merge_plan(rows, goal, experience, equipment, days, duration):
    all_details = [extract_details(row['description']) for _, row in rows.iterrows()]
    schedules = []
    exercises = []
    volume = {}

    for details in all_details:
        for item in details['schedule']:
            if item not in schedules:
                schedules.append(item)
        for item in details['prescriptions']:
            key = (item['exercise'].lower(), item['display'].lower())
            if not any((x['exercise'].lower(), x['display'].lower()) == key for x in exercises):
                exercises.append(item)
        for item in details['volume']:
            key = item['muscle_group'].lower()
            volume[key] = max(volume.get(key, 0), item['sets_per_week'])

    if len(schedules) > days:
        schedules = schedules[:days]
    elif not schedules:
        schedules = []

    if len(exercises) > max(6, days * 5):
        exercises = exercises[:max(6, days * 5)]

    return {
        'title': f'{goal} — Dataset Composite Plan',
        'goal': goal,
        'experience': experience,
        'equipment': equipment,
        'days': days,
        'duration': duration,
        'schedule': schedules,
        'exercises': exercises,
        'volume': [{'muscle_group': k.title(), 'sets_per_week': v} for k, v in volume.items()],
        'sources_used': len(rows),
    }


def estimate_avg_bpm(age, experience):
    pct = {'Beginner': 0.65, 'Intermediate': 0.75, 'Advanced': 0.85}[experience]
    return round((220 - age) * pct)


def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    if bmi < 25:
        return 'Normal'
    if bmi < 30:
        return 'Overweight'
    return 'Obese'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/input')
def input_page():
    return render_template('input.html', goals=GOALS, experience=EXPERIENCE, equipment=EQUIPMENT, days_options=DAYS_OPTIONS, duration_options=DURATION_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form.get('age', ''))
        height_cm = float(request.form.get('height_cm', ''))
        weight_kg = float(request.form.get('weight_kg', ''))
        days = int(request.form.get('days_per_week', ''))
        duration = int(request.form.get('session_minutes', ''))
    except ValueError:
        return render_template('input.html', goals=GOALS, experience=EXPERIENCE, equipment=EQUIPMENT, days_options=DAYS_OPTIONS, duration_options=DURATION_OPTIONS, errors=['Please enter valid values.'], form=request.form)

    gender = request.form.get('gender', '')
    goal = request.form.get('fitness_goal', '')
    experience = request.form.get('experience_level', '')
    equipment = request.form.get('equipment_access', '')

    if not (10 <= age <= 90 and 100 <= height_cm <= 230 and 30 <= weight_kg <= 250):
        return render_template('input.html', goals=GOALS, experience=EXPERIENCE, equipment=EQUIPMENT, days_options=DAYS_OPTIONS, duration_options=DURATION_OPTIONS, errors=['Age, height, or weight is outside the allowed range.'], form=request.form)
    if gender not in ('Male', 'Female') or goal not in GOALS or experience not in EXPERIENCE or equipment not in EQUIPMENT or days not in DAYS_OPTIONS or duration not in DURATION_OPTIONS:
        return render_template('input.html', goals=GOALS, experience=EXPERIENCE, equipment=EQUIPMENT, days_options=DAYS_OPTIONS, duration_options=DURATION_OPTIONS, errors=['Please select valid options.'], form=request.form)

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    avg_bpm = estimate_avg_bpm(age, experience)
    calorie_row = pd.DataFrame([{
        'Age': age, 'Weight (kg)': weight_kg, 'Height (m)': height_cm / 100,
        'BMI': bmi, 'Experience_Level': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}[experience],
        'Session_Duration (hours)': duration / 60,
        'Workout_Frequency (days/week)': days, 'Avg_BPM': avg_bpm, 'Gender': gender,
    }])[calorie_feature_order]
    calories = round(float(calorie_model.predict(calorie_row)[0]))

    rows = rank_programs(goal, experience, equipment, days, duration)
    plan = merge_plan(rows, goal, experience, equipment, days, duration)

    session['result'] = {
        'plan': plan,
        'predicted_calories': calories,
        'avg_bpm': avg_bpm,
        'age': age,
        'gender': gender,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': bmi,
        'bmi_category': bmi_category(bmi),
    }
    return redirect(url_for('result'))


@app.route('/result')
def result():
    data = session.get('result')
    if not data:
        return redirect(url_for('input_page'))
    return render_template('result.html', r=data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
