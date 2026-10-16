"""
대시보드에 표시하는 실험 결과 — 모두 노트북 실행 결과를 그대로 옮긴 값입니다.

- 후보 모델 비교 (CV_TABLE)   : Train 내부 CV만 사용 (02 baseline · 03 imbalance · 04 tuning · 05 feature engineering · 06 ensemble)
- 최종 모델 평가 (FINAL_...)  : 최종 모델 1개의 Test 389건 1회 평가 (07 final evaluation)
"""
import numpy as np
import pandas as pd

# ── 클래스 ──────────────────────────────────────────────────────────────────
CLASS_ORDER = ["Other_Faults", "Bumps", "K_Scratch", "Z_Scratch", "Pastry", "Stains", "Dirtiness"]
CONFUSED = {"Other_Faults", "Bumps", "Pastry"}
TEST_N = dict(zip(CLASS_ORDER, [135, 81, 78, 38, 32, 14, 11]))
TOTAL_N = dict(zip(CLASS_ORDER, [673, 402, 391, 190, 158, 72, 55]))
TRAIN_N = {c: TOTAL_N[c] - TEST_N[c] for c in CLASS_ORDER}

# ── 후보 모델 비교: Train 내부 CV만 사용 (Test 미사용) ───────────────────────
# 5-fold = 팀 공통 config.CV (StratifiedKFold 5, shuffle, random_state=42)
# 반복 CV = RepeatedStratifiedKFold(5×3, random_state=42)
_n = np.nan
CV_TABLE = pd.DataFrame(
    [
        # 모델, 단계, 피처, 불균형 처리, 5-fold Macro F1, std, 반복 CV, Weighted F1, Accuracy, Macro Precision, Macro Recall, ROC-AUC, 출처
        ["Logistic Regression", "① Baseline", "원본 26개", "없음", 0.7191, 0.0275, _n, 0.7104, 0.7132, 0.7324, 0.7156, 0.9308, "02"],
        ["Decision Tree", "① Baseline", "원본 26개", "없음", 0.7207, 0.0259, _n, 0.7052, 0.7075, 0.7218, 0.7288, 0.8367, "02"],
        ["Random Forest", "① Baseline", "원본 26개", "없음", 0.7899, 0.0312, _n, 0.7704, 0.7719, 0.8196, 0.7718, 0.9609, "02"],
        ["GBM", "① Baseline", "원본 26개", "없음", 0.7958, 0.0106, _n, 0.7775, 0.7784, 0.8125, 0.7850, 0.9584, "02"],
        ["XGBoost", "① Baseline", "원본 26개", "없음", 0.8109, 0.0308, _n, 0.7918, 0.7945, 0.8352, 0.7997, 0.9639, "02"],
        ["LightGBM", "① Baseline", "원본 26개", "없음", 0.8044, 0.0360, _n, 0.7875, 0.7893, 0.8266, 0.7936, 0.9606, "02"],
        ["Random Forest + SMOTENC", "② 불균형 처리", "원본 26개", "SMOTENC 균등화", 0.7955, 0.0150, 0.8024, 0.7727, 0.7719, 0.7872, 0.8118, _n, "03"],
        ["GBM + 클래스 가중치", "② 불균형 처리", "원본 26개", "CW α=1", 0.8095, 0.0088, 0.8089, 0.7829, 0.7822, 0.8085, 0.8187, _n, "03"],
        ["XGBoost + SMOTENC", "② 불균형 처리", "원본 26개", "SMOTENC 균등화", 0.8201, 0.0202, 0.8151, 0.8012, 0.8022, 0.8199, 0.8270, _n, "03"],
        ["LightGBM + SMOTENC", "② 불균형 처리", "원본 26개", "SMOTENC 균등화", 0.8268, 0.0240, 0.8209, 0.8032, 0.8035, 0.8370, 0.8230, _n, "03"],
        ["Random Forest (튜닝)", "③ 튜닝", "원본 26개", "탐색으로 선택", 0.7948, 0.0174, _n, 0.7676, 0.7674, 0.7837, 0.8159, _n, "04"],
        ["GBM (튜닝)", "③ 튜닝", "원본 26개", "CW α=1", 0.8225, 0.0182, 0.8207, 0.7969, 0.7964, 0.8258, 0.8251, _n, "04"],
        ["XGBoost (튜닝)", "③ 튜닝", "원본 26개", "CW α=1", 0.8226, 0.0230, 0.8211, 0.7933, 0.7932, 0.8247, 0.8269, _n, "04"],
        ["LightGBM (튜닝)", "③ 튜닝", "원본 26개", "SMOTENC 균등화", 0.8298, 0.0183, 0.8260, 0.8049, 0.8048, 0.8377, 0.8288, _n, "04"],
        ["Soft Voting (LGBM+XGB+GBM)", "③ 튜닝", "원본 26개", "구성 모델별", 0.8291, 0.0114, 0.8265, 0.8013, 0.8009, 0.8353, 0.8288, _n, "04"],
        ["LightGBM (파생변수 + 재튜닝)", "④ 파생변수", "36개", "SMOTENC + balanced", 0.8317, 0.0058, 0.8295, _n, _n, _n, _n, _n, "05"],
        ["Random Forest (36개)", "⑤ 재모델링·앙상블", "36개", "없음", 0.8109, 0.0202, _n, _n, _n, _n, _n, _n, "06"],
        ["ExtraTrees (36개)", "⑤ 재모델링·앙상블", "36개", "class_weight=balanced", 0.8180, 0.0185, _n, _n, _n, _n, _n, _n, "06"],
        ["XGBoost (36개)", "⑤ 재모델링·앙상블", "36개", "SMOTENC", 0.8297, 0.0202, _n, _n, _n, _n, _n, _n, "06"],
        ["LightGBM (36개)", "⑤ 재모델링·앙상블", "36개", "SMOTENC", 0.8329, 0.0160, _n, _n, _n, _n, _n, _n, "06"],
        ["GBM (36개)", "⑤ 재모델링·앙상블", "36개", "CW α=1", 0.8401, 0.0159, 0.8338, _n, _n, _n, _n, _n, "06"],
        ["Soft Voting (ET+GBM)", "⑤ 재모델링·앙상블", "36개", "구성 모델별", 0.8448, 0.0175, 0.8391, _n, _n, _n, _n, _n, "06"],
        ["Soft Voting (ET+RF+GBM) · 최종", "⑤ 재모델링·앙상블", "36개", "구성 모델별", 0.8441, 0.0188, 0.8389, _n, _n, _n, _n, _n, "06"],
    ],
    columns=["모델", "단계", "피처", "불균형 처리", "Macro F1 (5-fold)", "Macro F1 std", "Macro F1 (반복 CV 5×3)",
             "Weighted F1", "Accuracy", "Macro Precision", "Macro Recall", "Macro ROC-AUC", "노트북"],
)
STAGES = ["① Baseline", "② 불균형 처리", "③ 튜닝", "④ 파생변수", "⑤ 재모델링·앙상블"]
FINAL_CV_NAME = "Soft Voting (ET+RF+GBM) · 최종"

# 반복 CV(5×3)에서 '직전 최고 모델보다 좋은 fold 수' (05·06 노트북)
BETTER_FOLDS = {"Soft Voting (ET+GBM)": "11/15", FINAL_CV_NAME: "10/15", "GBM (36개)": "9/15",
                "LightGBM (파생변수 + 재튜닝)": "10/15"}

# ── 최종 모델 Test 1회 평가 (07 노트북) ──────────────────────────────────────
FINAL = dict(macro_f1=0.8324, accuracy=0.8098, precision=0.8486, recall=0.8182, weighted_f1=0.8123, roc_auc=0.9686,
             ci=(0.7904, 0.8686), p_ge_083=0.54, rep_cv=0.8389, rep_cv_std=0.0231, cv5=0.8441)

FINAL_PER_CLASS = pd.DataFrame(
    [(0.745, 0.800, 0.771), (0.756, 0.728, 0.742), (0.986, 0.923, 0.954), (1.000, 0.947, 0.973),
     (0.529, 0.562, 0.545), (0.923, 0.857, 0.889), (1.000, 0.909, 0.952)],
    columns=["Precision", "Recall", "F1"], index=CLASS_ORDER)

FINAL_CM = np.array([      # 행 = 실제, 열 = 예측 (CLASS_ORDER 순서)
    [108, 14, 1, 0, 11, 1, 0],
    [18, 59, 0, 0, 4, 0, 0],
    [5, 0, 72, 0, 1, 0, 0],
    [2, 0, 0, 36, 0, 0, 0],
    [10, 4, 0, 0, 18, 0, 0],
    [1, 1, 0, 0, 0, 12, 0],
    [1, 0, 0, 0, 0, 0, 10],
])

FINAL_KEY_ERRORS = pd.DataFrame(
    [("소수 유형 → Other_Faults·Bumps (놓침)", 17), ("Other_Faults → 소수 유형 (오경보)", 12),
     ("Bumps ↔ Other_Faults", 32), ("Other_Faults ↔ Pastry", 21), ("K_Scratch ↔ Z_Scratch", 0), ("전체 오분류", 74)],
    columns=["오류 종류", "건수"])

PERM_IMPORTANCE = pd.DataFrame(
    [("Steel_Plate_Thickness", 0.1702, 0.0126, False), ("Length_of_Conveyer", 0.1160, 0.0178, False),
     ("TypeOfSteel_A300", 0.1006, 0.0082, False), ("x_center_rel", 0.0244, 0.0022, True),
     ("X_Minimum", 0.0138, 0.0069, False), ("Edges_Y_Index", 0.0126, 0.0082, False),
     ("Minimum_of_Luminosity", 0.0109, 0.0031, False), ("Y_Maximum", 0.0099, 0.0034, False),
     ("compactness", 0.0099, 0.0043, True), ("X_Perimeter", 0.0096, 0.0030, False),
     ("perimeter_ratio_log", 0.0095, 0.0020, True), ("X_Maximum", 0.0093, 0.0059, False),
     ("x_straightness", 0.0091, 0.0053, True), ("Log_X_Index", 0.0077, 0.0037, False),
     ("Outside_Global_Index", 0.0073, 0.0036, False)],
    columns=["피처", "F1 감소량", "표준편차", "파생변수"],
)

IMBALANCE_CV = pd.DataFrame(
    [("처리 없음", 0.793, "기준"), ("클래스 가중치", 0.806, "10/15"), ("BorderlineSMOTE", 0.805, "11/15"),
     ("SMOTE", 0.810, "10/15"), ("언더샘플링", 0.673, "0/15")],
    columns=["방법", "Macro F1 (반복 CV 평균)", "처리 없음보다 좋은 fold"],
)

IMPROVEMENT_PATH = pd.DataFrame(
    [("Baseline GBM 기본값", 0.7958), ("불균형 처리: 클래스 가중치", 0.8095), ("하이퍼파라미터 튜닝", 0.8225),
     ("파생변수 10개 + 재튜닝", 0.8401), ("앙상블 ET+RF+GBM", 0.8441)],
    columns=["단계", "Macro F1 (5-fold CV)"],
)
IMPROVEMENT_PATH["직전 대비"] = IMPROVEMENT_PATH["Macro F1 (5-fold CV)"].diff()
