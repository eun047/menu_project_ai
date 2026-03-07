import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import numpy as np

# 데이터 로딩
with open("logs.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

# condition 모드
condition_logs = [log for log in logs if log["mode"] == "condition"]

meal_time_encoder = LabelEncoder()
menu_encoder_condition = LabelEncoder()

meal_times = [log["meal_time"] for log in condition_logs]
peoples = [int(log["people"]) for log in condition_logs]
menus_condition = [log["menu_name"] for log in condition_logs]
feedbacks_condition = [1 if log["feedback"] == "accepted" else 0 for log in condition_logs]

meal_times_encoded = meal_time_encoder.fit_transform(meal_times)
X_condition = np.column_stack([meal_times_encoded, peoples])
y_condition = np.array(feedbacks_condition)

model_condition = RandomForestClassifier(n_estimators=100, random_state=42)
model_condition.fit(X_condition, y_condition)

# tags 모드
tags_logs = [log for log in logs if log["mode"] == "tags"]

mlb = MultiLabelBinarizer()
menu_encoder_tags = LabelEncoder()

tags_list = [log["tags"] for log in tags_logs]
menus_tags = [log["menu_name"] for log in tags_logs]
feedbacks_tags = [1 if log["feedback"] == "accepted" else 0 for log in tags_logs]

X_tags = mlb.fit_transform(tags_list)
y_tags = np.array(feedbacks_tags)

model_tags = RandomForestClassifier(n_estimators=100, random_state=42)
model_tags.fit(X_tags, y_tags)

# 모델 저장
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model_condition": model_condition,
        "meal_time_encoder": meal_time_encoder,
        "menu_encoder_condition": menu_encoder_condition,
        "model_tags": model_tags,
        "mlb": mlb,
        "menu_encoder_tags": menu_encoder_tags
    }, f)

print("모델 학습 완료!")