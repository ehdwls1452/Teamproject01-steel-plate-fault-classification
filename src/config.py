from sklearn.model_selection import StratifiedKFold


# =========================
# 공통 설정
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]


# =========================
# Cross Validation
# =========================

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# =========================
# 평가 지표
# =========================

SCORING = {
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "accuracy": "accuracy",
}