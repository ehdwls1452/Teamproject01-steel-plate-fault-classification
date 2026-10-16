"""
강판 표면 결함 분류 — 보고서용 추가 실험 (박태상)

최종 모델 재현은 run_ensemble_taesang.py 를 쓴다. 이 파일은 보고서에 실린
누수 점검·과적합 검증·기각 실험 등 부가 분석을 모아둔 연구용 스크립트다.
---------------------------------------------------------------
VS Code 주피터 노트북에서는 아래 '# %%' 구분선 단위로 셀을 나눠 실행하면 됩니다.
터미널에서 통째로 돌릴 때: uv run python src/experiments_taesang.py
"""

# %% ── 0. 준비 ────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict, train_test_split)
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from scipy.stats import kruskal
import lightgbm as lgb
from xgboost import XGBClassifier

from features_taesang import add_features
from validation_taesang import make_coil_groups

BASE_DIR = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                if (p / "src" / "config.py").exists())
RAW = BASE_DIR / "data" / "raw" / "Faults.csv"
PROC = BASE_DIR / "data" / "processed" / "Faults_processed.csv"
SEED = 42


# %% ── 1. 전처리: 라벨 7열 → 1열, 중복 강종 변수 제거 ───────────
def build_processed():
    raw = pd.read_csv(RAW)
    labels = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
              "Dirtiness", "Bumps", "Other_Faults"]

    # (검증) 한 행에 라벨이 정확히 하나만 1인가? -> 다중분류가 맞는지 확인
    assert (raw[labels].sum(axis=1) == 1).all(), "다중라벨 데이터임. 설계 재검토 필요"
    # (검증) 강종 두 변수는 서로 배타적인가? -> 하나는 중복 정보
    assert (raw.TypeOfSteel_A300 + raw.TypeOfSteel_A400 == 1).all()

    df = raw.drop(columns=labels + ["TypeOfSteel_A400"]).copy()
    df["target"] = raw[labels].idxmax(axis=1).replace({"K_Scatch": "K_Scratch"})
    PROC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC, index=False)
    return df


df = pd.read_csv(PROC) if PROC.exists() else build_processed()
y_raw = df["target"]
X0 = df.drop(columns=["target"])

print(df.shape, "| 결측:", df.isna().sum().sum(), "| 중복:", df.duplicated().sum())
print(y_raw.value_counts())


# %% ── 2. EDA: 변수별 클래스 분리력 (Kruskal-Wallis) ───────────
# 분포가 치우치고 극단값이 많아 분산분석 대신 순위 기반 검정을 사용
def separability(X, y):
    rows = []
    for col in X.columns:
        samples = [X.loc[y == c, col].values for c in y.unique()]
        H, p = kruskal(*samples)
        k, n = len(samples), len(X)
        rows.append((col, H, p, (H - k + 1) / (n - k)))   # eta^2 근사
    return (pd.DataFrame(rows, columns=["feature", "H", "p", "eta2"])
              .sort_values("eta2", ascending=False))


print(separability(X0, y_raw).head(12).to_string(index=False))


# %% ── 3. 누수 점검: 결함은 강판 위에 뭉쳐서 생기는가 ───────────
def leakage_check(df, gap=50_000):
    d = df.sort_values(["Length_of_Conveyer", "Y_Minimum"]).reset_index(drop=True)
    same = (d.Length_of_Conveyer == d.Length_of_Conveyer.shift())
    near = same & ((d.Y_Minimum - d.Y_Minimum.shift()).abs() < gap)
    observed = (d.target[near] == d.target.shift()[near]).mean()
    expected = (df.target.value_counts(normalize=True) ** 2).sum()
    return near.sum(), observed, expected


n, obs, exp = leakage_check(df)
print(f"인접 결함 쌍 {n}건 | 종류 일치율 {obs:.3f} | 무작위 기댓값 {exp:.3f}")
# 관측치가 기댓값보다 크게 높으면 -> 무작위 분할은 성능을 과대평가함


# %% ── 4. 파생변수 + 입력 행렬 구성 ───────────────────────────
Xf = add_features(X0)
SEL = ["f_width", "f_height", "f_aspect", "f_bbox_area", "f_fill_rate", "f_compact",
       "f_perim_ratio", "f_rough_x", "f_rough_y", "f_perim_area", "f_elong",
       "f_thick_x_type", "f_area_per_thick", "f_lum_mean", "f_lum_contrast",
       "f_lum_density", "f_xpos_rel", "f_edge_dist", "f_width_rel"]
COLS = list(X0.columns) + SEL

le = LabelEncoder()
y = le.fit_transform(y_raw)
X = Xf[COLS].values.astype(float)
CLASSES = le.classes_
groups = make_coil_groups(df)          # 코일 단위 분할용 그룹 ID
print("입력 변수:", X.shape[1], "| 추정 코일 수:", len(np.unique(groups)))


# %% ── 5. 모델 정의 ───────────────────────────────────────────
# 트리 계열: 스케일링 불필요 (값의 순서만 보고 분기하므로)
# 선형/거리 계열: QuantileTransformer 필수 (극단값이 많아 표준화는 부적절)
def tree_models(seed=SEED):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=900, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=900, max_features=0.5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=900, learning_rate=0.04, max_depth=7, subsample=0.9,
            colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
            random_state=seed, n_jobs=-1, eval_metric="mlogloss"),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=900, learning_rate=0.03, num_leaves=31,
            min_child_samples=10, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
            random_state=seed, verbose=-1, n_jobs=-1),
    }


def scaled_models():
    qt = lambda: QuantileTransformer(output_distribution="normal",
                                     n_quantiles=500, random_state=0)
    return {
        "LogReg": make_pipeline(qt(), LogisticRegression(
            max_iter=3000, C=2.0, class_weight="balanced")),
        "SVM": make_pipeline(qt(), SVC(C=10, gamma="scale", class_weight="balanced")),
        "MLP": make_pipeline(qt(), MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=700, random_state=0)),
    }


ENSEMBLE = ["LightGBM", "XGBoost", "ExtraTrees"]   # 계열이 서로 다른 셋


# %% ── 6. 반복 교차검증 (주 평가) ──────────────────────────────
# 데이터가 2천 건 아래라 1회 분할은 운에 좌우됨 -> 시드 3개 평균으로 보고
def repeated_cv(names, n_repeats=3):
    probs = {}
    for r in range(n_repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=100 + r)
        for nm in names:
            m = tree_models(seed=100 + r)[nm]
            probs[(r, nm)] = cross_val_predict(m, X, y, cv=cv, method="predict_proba")
            print(f"  rep{r} {nm} 완료", flush=True)
    return probs


def summarize(probs, combos, n_repeats=3):
    rows = []
    for combo in combos:
        m, w = [], []
        for r in range(n_repeats):
            p = np.mean([probs[(r, c)] for c in combo], axis=0).argmax(1)
            m.append(f1_score(y, p, average="macro"))
            w.append(f1_score(y, p, average="weighted"))
        rows.append(["+".join(combo), np.mean(m), np.std(m), np.mean(w)])
    return pd.DataFrame(rows, columns=["model", "macro_F1", "std", "weighted_F1"])


ALL = ["RandomForest", "ExtraTrees", "HistGB", "XGBoost", "LightGBM"]
probs = repeated_cv(ALL)
combos = [(n,) for n in ALL] + [tuple(ENSEMBLE)]
print(summarize(probs, combos).to_string(index=False))


# %% ── 7. 최종 앙상블 평가 ────────────────────────────────────
P = np.mean([probs[(r, n)] for r in range(3) for n in ENSEMBLE], axis=0)
pred = P.argmax(1)
print(classification_report(y, pred, target_names=CLASSES, digits=3))
print(pd.DataFrame(confusion_matrix(y, pred), index=CLASSES, columns=CLASSES))


# %% ── 8. 코일 단위 분할 검증 (현장 재현 성능) ──────────────────
def compare_splits():
    out = {}
    for label, cv, g in [
        ("무작위 5-fold", StratifiedKFold(5, shuffle=True, random_state=SEED), None),
        ("코일단위 5-fold", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), groups),
    ]:
        Ps = [cross_val_predict(tree_models()[n], X, y, cv=cv, groups=g,
                                method="predict_proba") for n in ENSEMBLE]
        p = np.mean(Ps, axis=0).argmax(1)
        out[label] = dict(macro=f1_score(y, p, average="macro"),
                          weighted=f1_score(y, p, average="weighted"),
                          acc=(p == y).mean())
    return pd.DataFrame(out).T


print(compare_splits())


# %% ── 9. 변수 중요도 (순열 기반) ─────────────────────────────
# 트리 기본 중요도는 값 종류가 많은 변수를 과대평가하므로 순열 중요도를 사용
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2,
                            stratify=y, random_state=SEED)
m = tree_models()["LightGBM"].fit(X[itr], y[itr])
from sklearn.metrics import make_scorer
r = permutation_importance(m, X[ite], y[ite], n_repeats=10, random_state=0, n_jobs=-1,
                           scoring=make_scorer(f1_score, average="macro"))
print(pd.Series(r.importances_mean, index=COLS).sort_values(ascending=False).head(15))


# %% ── 10. 성능의 출처 분해 ───────────────────────────────────
# 공정 조건 변수만으로도 높은 점수가 나오면, 모델이 '제품 규격 분류기'로
# 작동하고 있다는 뜻 -> 제품 구성이 바뀌면 성능이 깨질 수 있음
PROC_VARS = ["Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300",
             "f_thick_x_type", "f_area_per_thick"]
pi = [COLS.index(c) for c in PROC_VARS]
keep = [i for i in range(len(COLS)) if i not in pi]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, cols in [("전체 변수", list(range(len(COLS)))),
                   ("공정조건 제외", keep),
                   ("공정조건만", pi)]:
    p = cross_val_predict(tree_models()["LightGBM"], X[:, cols], y, cv=cv)
    print(f"{name:12s} ({len(cols):2d}개) macro_F1={f1_score(y, p, average='macro'):.4f}")


# %% ── 11. 과적합 검증 ① 학습곡선 ──────────────────────────────
# 학습량을 늘려가며 '학습 점수'와 '검증 점수'를 함께 기록한다.
# 판단 기준: 검증 점수가 계속 오르는가 / 두 점수의 차이와 폴드 간 편차가 줄어드는가
def learning_curve(fracs=(0.15, 0.3, 0.45, 0.6, 0.8, 1.0), seed=0):
    rng, cv, rows = np.random.default_rng(seed), StratifiedKFold(5, shuffle=True, random_state=SEED), []
    for f in fracs:
        tr_s, va_s = [], []
        for tr, va in cv.split(X, y):
            idx = np.concatenate([rng.choice(tr[y[tr] == c],
                                             max(3, int(round((y[tr] == c).sum() * f))),
                                             replace=False) for c in np.unique(y)])
            Ptr, Pva = [], []
            for nm in ENSEMBLE:
                m = tree_models()[nm].fit(X[idx], y[idx])
                Ptr.append(m.predict_proba(X[idx])); Pva.append(m.predict_proba(X[va]))
            tr_s.append(f1_score(y[idx], np.mean(Ptr, 0).argmax(1), average="macro"))
            va_s.append(f1_score(y[va], np.mean(Pva, 0).argmax(1), average="macro"))
        rows.append([len(idx), np.mean(tr_s), np.mean(va_s), np.std(va_s)])
    return pd.DataFrame(rows, columns=["n", "train_F1", "valid_F1", "valid_std"])


print(learning_curve().to_string(index=False))


# %% ── 12. 과적합 검증 ② 복잡도 비교 · 라벨 섞기 ────────────────
# 복잡도를 낮춰서 좋아지면 과적합 구간, 라벨을 섞었는데 점수가 남으면 누수
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for nl, mcs, rl in [(7, 40, 20), (15, 20, 5), (31, 10, 1), (63, 5, 1), (127, 2, 0)]:
    m = tree_models()["LightGBM"].set_params(num_leaves=nl, min_child_samples=mcs, reg_lambda=rl)
    p = cross_val_predict(m, X, y, cv=cv)
    print(f"leaves={nl:3d} -> valid macro_F1={f1_score(y, p, average='macro'):.4f}")

rng = np.random.default_rng(0)
sh = [f1_score(ys := rng.permutation(y),
               cross_val_predict(tree_models()["LightGBM"], X, ys,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=i)),
               average="macro") for i in range(3)]
print(f"라벨 섞음 macro_F1 = {np.mean(sh):.4f} (무작위 찍기 ≈ {1/len(CLASSES):.3f})")


# %% ── 13. 판정 임계값 조정 (중첩 교차검증) ─────────────────────
# 배율을 '정할 때 보지 않은' 데이터로만 평가해야 공정하다
def tune_weights(P, yy, targets, grid=np.linspace(0.5, 3.0, 26), rounds=12):
    w = np.ones(P.shape[1]); best = f1_score(yy, (P * w).argmax(1), average="macro")
    for _ in range(rounds):
        improved = False
        for c in targets:
            cur = w[c]
            for g in grid:
                w[c] = g
                sc = f1_score(yy, (P * w).argmax(1), average="macro")
                if sc > best + 1e-6: best, cur, improved = sc, g, True
            w[c] = cur
        if not improved: break
    return w


def nested_threshold_test():
    idx = {c: i for i, c in enumerate(CLASSES)}
    pb = [idx["Pastry"], idx["Bumps"]]
    out = {"보정 없음": [], "Pastry·Bumps만": [], "7개 클래스 전부": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        oof = np.zeros((len(tr), len(CLASSES)))
        for i_tr, i_va in StratifiedKFold(4, shuffle=True, random_state=1).split(X[tr], y[tr]):
            oof[i_va] = tree_models()["LightGBM"].fit(X[tr][i_tr], y[tr][i_tr]).predict_proba(X[tr][i_va])
        Pte = tree_models()["LightGBM"].fit(X[tr], y[tr]).predict_proba(X[te])
        out["보정 없음"].append(f1_score(y[te], Pte.argmax(1), average="macro"))
        for key, tg in [("Pastry·Bumps만", pb), ("7개 클래스 전부", range(len(CLASSES)))]:
            w = tune_weights(oof, y[tr], tg)
            out[key].append(f1_score(y[te], (Pte * w).argmax(1), average="macro"))
    return pd.DataFrame({k: [np.mean(v), np.std(v)] for k, v in out.items()},
                        index=["mean", "std"]).T


print(nested_threshold_test())


# %% ── 14. 라벨 체계가 만든 천장 확인 ───────────────────────────
# "기타 결함"은 여러 결함을 몰아넣은 범주이므로, 빼고 재학습해 한계를 측정한다
oi = list(CLASSES).index("Other_Faults")
m6 = y != oi
y6 = np.unique(y[m6], return_inverse=True)[1]
for name, (XX, yy) in {"7종 (포함)": (X, y), "6종 (제외)": (X[m6], y6)}.items():
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    P = np.mean([cross_val_predict(tree_models()[n], XX, yy, cv=cv, method="predict_proba")
                 for n in ENSEMBLE], 0)
    print(f"{name:12s} macro_F1={f1_score(yy, P.argmax(1), average='macro'):.4f} "
          f"acc={(P.argmax(1) == yy).mean():.4f}")
