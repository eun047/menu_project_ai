from flask import Flask, render_template, request, redirect
import os
import json
import random
import pickle
import numpy as np

app = Flask(__name__)

# 데이터 로딩
def load_all_menus(data_dir = "data"):
    all_menus = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                menus = data.get("menus", [])
                all_menus.extend(menus)

    return all_menus

# 모델 로딩
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    return None

# 태그 목록 수집
def collect_all_tags(menus):
    tag_set = set()

    for menu in menus:
        for tag in menu["tags"]:
            tag_set.add(tag)
    
    return sorted(tag_set)


# 상황 기반 추천 로직
def recommend_by_condition(menus, meal_time, people):
    candidates = []

    for menu in menus:
        if meal_time in menu["meal_time"]:
            if menu["min_people"] <= people <= menu["max_people"]:
                candidates.append(menu)

    if not candidates:
        return None

    # 모델 적용
    model_data = load_model()
    if model_data:
        try:
            model = model_data["model_condition"]
            meal_time_encoder = model_data["meal_time_encoder"]

            meal_time_encoded = meal_time_encoder.transform([meal_time])[0]
            scores = []

            for candidate in candidates:
                X = np.array([[meal_time_encoded, people]])
                score = model.predict_proba(X)[0][1]  # accepted 확률
                scores.append(score)

            # 점수 높은 메뉴 반환
            best_index = scores.index(max(scores))
            return candidates[best_index]
        except:
            pass

    # 모델 없으면 랜덤
    return random.choice(candidates)

# 태그 기반 추천 로직 (하나라도 포함되면 후보)
def recommend_by_tags(menus, selected_tags):
    candidates = []

    for menu in menus:
        if any(tag in menu["tags"] for tag in selected_tags):
            candidates.append(menu)

    if not candidates:
        return None

    return random.choice(candidates)

# 메인 페이지 (방식 선택)
@app.route("/")
def index():
    return render_template("index.html")


# 상황 선택 페이지
@app.route("/condition")
def condition_page():
    return render_template("condition.html")


# 태그 선택 페이지
@app.route("/tags")
def tags_page():
    menus = load_all_menus()
    all_tags = collect_all_tags(menus)
    return render_template("tags.html", tags=all_tags)


# 결과 페이지
@app.route("/result", methods=["POST"])
def result_page():
    menus = load_all_menus()
    result = None
    mode = request.form.get("mode")
    feedback = request.form.get("feedback")
    prev_menu = request.form.get("prev_menu")

    # rejected 로그 저장
    if feedback == "rejected" and prev_menu:
        log_entry = {
            "menu_name": prev_menu,
            "mode": mode,
            "feedback": "rejected",
            "meal_time": request.form.get("meal_time"),
            "people": request.form.get("people"),
            "tags": request.form.getlist("tags")
        }

        logs_path = "logs.json"
        if os.path.exists(logs_path):
            with open(logs_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    if mode == "condition":
        meal_time = request.form.get("meal_time")
        people = int(request.form.get("people", 1))
        result = recommend_by_condition(menus, meal_time, people)
        return render_template("result.html", result=result, mode=mode,
                               meal_time=meal_time, people=people)

    elif mode == "tags":
        selected_tags = request.form.getlist("tags")
        result = recommend_by_tags(menus, selected_tags)
        return render_template("result.html", result=result, mode=mode,
                               selected_tags=selected_tags)

    return render_template("result.html", result=None)

# 피드백 저장
@app.route("/feedback", methods=["POST"])
def feedback():
    menu_name = request.form.get("menu_name")
    mode = request.form.get("mode")
    feedback = request.form.get("feedback")
    meal_time = request.form.get("meal_time")
    people = request.form.get("people")
    selected_tags = request.form.getlist("tags")

    log_entry = {
        "menu_name": menu_name,
        "mode": mode,
        "feedback": feedback,
        "meal_time": meal_time,
        "people": people,
        "tags": selected_tags
    }

    # logs.json에 저장
    logs_path = "logs.json"
    if os.path.exists(logs_path):
        with open(logs_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return redirect("/")

# 실행
if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0", port=int(os.environ.get("PORT", 10000)))
