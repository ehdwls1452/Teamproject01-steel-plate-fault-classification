"""
최종 앙상블 (LightGBM + XGBoost + ExtraTrees) 재현 스크립트 — 박태상

팀 공용 규약을 그대로 따른다.
  - 데이터: load_data()  (공용 Train/Test 인덱스 사용)
  - 평가  : config.CV    (Train에 대한 5겹 교차검증)
  - 기록  : evaluate.log_result()  →  results/model_scores.csv

실행:  uv run python src/run_ensemble_taesang.py
필요 패키지:  uv add lightgbm xgboost
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                if (p / "src" / "config.py").exists())
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import lightgbm as lgb
from xgboost import XGBClassifier

from config import CV, RANDOM_STATE as R
from load_data import load_data
from evaluate import log_result, results_table
from features_taesang import add_features

AUTHOR = "박태상"

# 6.4절 전진 비교로 채택한 파생변수 19개
SELECTED = [
    "f_width", "f_height", "f_aspect", "f_bbox_area", "f_fill_rate", "f_compact",
    "f_perim_ratio", "f_rough_x", "f_rough_y", "f_perim_area", "f_elong",
    "f_thick_x_type", "f_area_per_thick", "f_lum_mean", "f_lum_contrast",
    "f_lum_density", "f_xpos_rel", "f_edge_dist", "f_width_rel",
]


def build_features(X):
    """원본 26변수 + 파생 19변수 = 45변수"""
    return add_features(X)[list(X.columns) + SELECTED]


def members(seed=R):
    """계열이 서로 다른 세 모델. 같은 계열을 여러 개 넣으면 묶는 의미가 없다."""
    return [
        lgb.LGBMClassifier(
            n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=10,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced", random_state=seed, verbose=-1, n_jobs=-1),
        XGBClassifier(
            n_estimators=900, learning_rate=0.04, max_depth=7, subsample=0.9,
            colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
            random_state=seed, n_jobs=-1, eval_metric="mlogloss"),
        ExtraTreesClassifier(
            n_estimators=900, max_features=0.5, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1),
    ]


def soft_vote_cv(X, y):
    """세 모델의 교차검증 예측 확률을 같은 가중으로 평균한다.

    검증 점수에 맞춰 가중치를 조정하면 그 점수에만 유리해지므로 동일 가중을 쓴다.
    """
    classes = sorted(y.unique())
    code = {c: i for i, c in enumerate(classes)}
    yi = y.map(code).values

    probs = [cross_val_predict(m, X.values, yi, cv=CV, method="predict_proba")
             for m in members()]
    avg = np.mean(probs, axis=0)
    return pd.Series([classes[i] for i in avg.argmax(1)], index=y.index)


def main():
    X_train, X_test, y_train, y_test = load_data()
    Xa = build_features(X_train)
    print(f"학습 데이터 {Xa.shape[0]}건 / 변수 {Xa.shape[1]}개 "
          f"(원본 {X_train.shape[1]} + 파생 {len(SELECTED)})\n")

    pred = soft_vote_cv(Xa, y_train)

    log_result(
        name="LGBM+XGB+ET_SoftVoting_파생변수",
        y_true=y_train,
        y_pred=pred,
        author=AUTHOR,
        note="클래스 가중치, 파생변수 19개, 계열이 다른 3종 소프트보팅",
    )

    print("\n[결함 종류별 성능]")
    print(classification_report(y_train, pred, digits=3))

    print("[누적 결과]")
    print(results_table().to_string(index=False))

    print("\n* Test 세트는 최종 제출 직전까지 사용하지 않는다.")


if __name__ == "__main__":
    main()
