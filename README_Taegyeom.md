# 강판 결함 유형 분류 — 김태겸 작업 정리

Steel Plates Faults(UCI) 데이터의 측정값 27개로 강판 결함 7가지 유형을 분류하는 다중 분류 모델입니다.
핵심 지표는 **Macro F1-score**(목표 ≥ 0.83)입니다.

## 최종 모델

| 항목 | 내용 |
|---|---|
| 모델 | **Soft Voting (ExtraTrees + Random Forest + GBM)** — 세 모델의 예측 확률 평균 |
| 입력 피처 | 원본 26개(TypeOfSteel_A400 제외) + 파생변수 10개 = 36개 |
| 불균형 처리 | 샘플러 없음, 클래스 가중치 (ExtraTrees `class_weight='balanced'`, GBM 균형 `sample_weight` α=1) |
| 반복 층화 CV(5×3) Macro F1 | **0.8395 ± 0.024** (모델 선택 기준) |
| Test Macro F1 | **0.828** (95% CI 0.785 ~ 0.865), Accuracy 0.805 |
| 모델 파일 | `models/Taegyeom_final_model.pkl` |
| 설정 | `results/final/Taegyeom_final_model_config.json` |

## 진행 과정 (Macro F1)

| 단계 | 노트북 | 결과 |
|---|---|---|
| 1. EDA·전처리 | `Taegyeom_01_eda_preprocessing.ipynb` | A400 제외(A300과 완전 중복), 팀 공통 전처리와 일치 검증 |
| 2. Baseline | `Taegyeom_02_baseline.ipynb` | 6개 모델 비교, 최고 XGBoost 5-fold 0.811 |
| 3. 불균형 처리 | `Taegyeom_03_imbalance.ipynb` | 가중치·언더샘플링·SMOTENC·SMOTE 비교, LightGBM + SMOTENC 0.827 (5-fold) |
| 4. 튜닝 | `Taegyeom_04_tuning.ipynb` | RandomizedSearchCV, 튜닝 LightGBM 반복 CV 0.8260 |
| 5. 파생변수 | `Taegyeom_05_feature_engineering.ipynb` | 파생변수 10개 + 재튜닝, 반복 CV 0.8295 |
| 6. 앙상블 | `Taegyeom_06_ensemble.ipynb` | 36개 피처로 5개 모델 재튜닝 + Soft Voting·Stacking 비교, **반복 CV 0.8395** |
| 7. 최종 평가 | `Taegyeom_07_final_evaluation.ipynb` | Train 전체 재학습 → Test 평가, 클래스별 지표·혼동행렬·오분류 사례·피처 중요도 |

## 파생변수 10개

- 형상: `compactness`, `perimeter_ratio_log`, `x_straightness`
- 밝기: `lum_range`, `lum_contrast`, `lum_mean_position`
- 위치: `x_center_rel`, `x_center_dist`
- 강판 맥락: `plate_defect_count`(같은 강판 조건의 결함 수, 학습 fold에서만 집계), `thickness_per_conveyer`

행 단위 8개는 `add_features()`, 강판 맥락 2개는 Pipeline 안의 `PlateContextEncoder`가 만듭니다(`src/Taegyeom_model_utils.py`).

## 실행 방법

```python
# notebooks 폴더에서
import sys; sys.path.insert(0, "../src")
import joblib
from load_data import load_data
from Taegyeom_model_utils import add_features

X_train, X_test, y_train, y_test = load_data()
model = joblib.load("../models/Taegyeom_final_model.pkl")
pred = model.predict(add_features(X_test))      # 입력: add_features 적용 데이터
proba = model.predict_proba(add_features(X_test))
```

- 노트북은 01 → 07 순서로 실행합니다. 튜닝·앙상블 결과는 `results/*/Taegyeom_*_cache.pkl`에 저장되어 있어 `RERUN = False`(기본값)면 몇 분 안에 끝납니다.
- 필요 패키지: scikit-learn, imbalanced-learn, lightgbm, xgboost, pandas, matplotlib, joblib

## 검증 원칙

- Train/Test 80:20 층화 분할(random_state=42), Train 내부 StratifiedKFold 5-fold(shuffle, 42)
- 전처리·샘플링·가중치·강판 맥락 변수는 모두 Pipeline 안에서 학습 fold로만 계산
- 모델 비교는 반복 층화 CV(5×3) 평균 기준 — 5-fold 튜닝 최고값은 여러 설정 중 1등을 고른 값이라 낙관적
- Test는 단계별 최고 모델을 1회씩, 총 3회 평가(26개 LightGBM 0.825 → 36개 LightGBM 0.823 → Soft Voting 0.828). 모델 갱신은 CV 결과만으로 판단

## 한계

- Pastry가 가장 약한 유형(Test F1 0.531)
- 오분류 76건 중 35건이 Bumps ↔ Other_Faults 혼동
- Train에 없던 강판 조건의 결함은 오분류율 33%(전체 20%)
- 희소 클래스 Test 건수가 적어(Dirtiness 11건, Stains 14건) 클래스별 지표 불확실성이 큼

## 폴더 구성

```
notebooks/  Taegyeom_01 ~ 07 노트북
src/        Taegyeom_model_utils.py (전처리·Pipeline·평가·파생변수 함수)
results/    eda, baseline, imbalance, tuning, feature_engineering, ensemble, final
figures/    Taegyeom_*.png
models/     Taegyeom_final_model.pkl
docs/       Taegyeom_모델링_프로젝트_보고서.docx
```
