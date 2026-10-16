"""
강판 결함 유형 분류 — 결과 공유용 Streamlit 대시보드

실행:  streamlit run app.py

파일 3개를 같은 폴더(프로젝트 루트)에 둡니다.
    app.py               화면 6개 (이 파일)
    dashboard_data.py    노트북에서 옮긴 실험 결과 수치
    dashboard_utils.py   경로 · 스타일 · 표/차트 · 데이터와 모델 불러오기 · 예측

프로젝트 폴더 구조
    src/        load_data.py, config.py, Taegyeom_model_utils.py   ← 저장된 모델을 불러올 때 필요
    data/       processed/Faults_processed.csv, split/train_idx.csv, split/test_idx.csv
    figures/    분석 노트북에서 저장한 png
    results/    model_scores.csv
    models/     Taegyeom_final_model_trained.pkl (또는 Taegyeom_final_model.pkl)

대시보드는 모델을 다시 학습하지 않습니다. 저장된 모델은 '예측 체험'과 '결과 재검증'에서 예측에만 씁니다.
"""
import traceback

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="강판 결함 분류 대시보드", page_icon="🔩", layout="wide")

from dashboard.dashboard_data import (BETTER_FOLDS, CLASS_ORDER, CONFUSED, CV_TABLE, FINAL, FINAL_CM, FINAL_CV_NAME,  # noqa: E402
                            FINAL_KEY_ERRORS, FINAL_PER_CLASS, IMBALANCE_CV, IMPROVEMENT_PATH, PERM_IMPORTANCE,
                            STAGES, TEST_N, TOTAL_N, TRAIN_N)
from dashboard.dashboard_utils import (BLUE, BLUE_LIGHT, GRAY, GREEN, MUTED, ORANGE, PLATE_KEYS, RED,  # noqa: E402
                             apply_style, caption, chart, confusion_chart, figure, hbar, kpis, load_model,
                             load_scores, load_split, note, page_header, predict_proba, show_error, table)

apply_style()


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터
# ══════════════════════════════════════════════════════════════════════════════
def page_data():
    page_header("데이터", "강판 표면 결함 1,941건을 7개 유형으로 분류합니다. 유형별 건수 차이가 크지만, "
                       "건수가 적은 유형이 반드시 맞히기 어려운 것은 아니라는 점이 이후 모델링 방향을 정했습니다.")
    kpis([("전체 결함", "1,941건", "Train 1,552 / Test 389 (층화 분할)"),
          ("결함 유형", "7개", "다중 분류"),
          ("최다 ÷ 최소", "12.2배", "Other_Faults 673건 vs Dirtiness 55건"),
          ("입력 변수", "26개", "위치·형상·밝기·공정 조건")])
    figure("target_distribution.png",
           "① 유형별 건수(진한 색 Train, 연한 색 Test) ② 유형별 건수와 Baseline XGBoost F1(Train 5-fold CV)의 관계")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.subheader("유형별 건수")
        counts = pd.DataFrame({"유형": CLASS_ORDER,
                               "Train": [TRAIN_N[c] for c in CLASS_ORDER],
                               "Test": [TEST_N[c] for c in CLASS_ORDER],
                               "전체": [TOTAL_N[c] for c in CLASS_ORDER]})
        counts["비율(%)"] = counts["전체"] / counts["전체"].sum() * 100
        counts["구분"] = ["서로 헷갈리는 유형" if c in CONFUSED else "잘 구분되는 유형" for c in CLASS_ORDER]
        table(counts, {"Train": "%d", "Test": "%d", "전체": "%d", "비율(%)": "%.1f"})
    with right:
        st.subheader("해석")
        st.markdown(
            "- **잘 구분되는 유형**(K_Scratch, Z_Scratch, Stains, Dirtiness)은 건수가 적어도 F1이 0.85 이상입니다.\n"
            "- **서로 헷갈리는 유형**(Other_Faults, Bumps, Pastry)은 F1이 0.8 미만이며, Other_Faults는 가장 많은데도 어렵습니다.\n"
            "- 그래서 단순히 소수 유형을 늘리는 것보다 **헷갈리는 세 유형의 경계를 다듬는 것**이 성능에 더 중요했습니다.\n"
            "- 평가 지표는 유형별 F1을 똑같이 평균하는 **Macro F1**을 기준으로 삼았습니다.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 데이터 구조
# ══════════════════════════════════════════════════════════════════════════════
def page_structure():
    page_header("데이터 구조", "결함은 서로 독립적이지 않습니다. 같은 생산 조건(강판 폭·두께·강종)으로 묶이면 같은 유형이 반복되고, "
                            "롤 자국처럼 일정한 주기로 찍힌 결함도 있습니다.")
    tab1, tab2 = st.tabs(["생산 조건 묶음 구조", "롤 마크 주기"])
    with tab1:
        kpis([("같은 조건 안 최다 유형 비율", "0.763", "정답을 무작위로 섞으면 평균 0.431"),
              ("처음 보는 조건의 오분류율", "29.2%", "Test 24건 기준"),
              ("이미 본 조건의 오분류율", "18.4%", "Test 365건 기준")], accent_first=True)
        figure("process_condition.png",
               "① 공정 조건 묶음 안의 유형 집중도 ② 강종(A300/A400)별 유형 구성 ③ 유형별 강판 폭 범위 ④ 처음 보는 조건에서의 오분류율")
        st.markdown(
            "- **강종만으로도 일부 유형은 거의 결정됩니다.** K_Scratch는 A300이 0%, Z_Scratch는 91%입니다.\n"
            "- **Stains는 사실상 한 가지 강판 폭(범위 8)에서만 나왔습니다.** 모델이 결함 모양보다 생산 조건을 외울 위험이 있습니다.\n"
            "- 이 구조 때문에 최종 모델에서도 두께·컨베이어 길이·강종이 중요도 1~3위를 차지합니다(최종 모델 평가 화면 참고).")
    with tab2:
        kpis([("한 강판 묶음의 K_Scratch", "92건", "폭 1358 · 두께 40 · A400, 왼쪽 가장자리"),
              ("기본 주기", "3,548", "Y_Minimum 기준 간격"),
              ("주기의 정수배와 맞는 간격", "81%", "무작위 배치 8% / 1%")], accent_first=True)
        figure("roll_mark_period.png",
               "① 한 강판 묶음의 K_Scratch 위치 ② 이웃 결함 간격 ÷ 기본 주기 ③ 실제 데이터와 무작위 배치의 규칙성 비교")
        note("롤 자국은 같은 강판 안에서 일정 간격으로 반복되므로 Train과 Test에 같은 강판의 결함이 나뉘어 들어갑니다. "
             "일반 CV 점수가 실제 신규 강판에서보다 높게 나올 수 있는 이유이며, 일반화 점검 화면에서 그룹 CV로 확인합니다.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모델 비교 (Train 내부 CV만 사용)
# ══════════════════════════════════════════════════════════════════════════════
def page_models():
    page_header("모델 비교", "모든 후보 비교는 Train 내부 교차검증(CV) 결과로만 합니다. Test 데이터는 후보 비교에 쓰지 않고, "
                          "최종 모델 1개를 평가할 때 한 번만 씁니다(최종 모델 평가 화면).")
    tabs = st.tabs(["단계별 개선", "불균형 처리 전후", "후보 모델 비교 (CV)", "최종 모델 선정 순위", "팀 실험 비교"])
    with tabs[0]:
        models_improvement()
    with tabs[1]:
        models_imbalance()
    with tabs[2]:
        models_candidates()
    with tabs[3]:
        models_ranking()
    with tabs[4]:
        models_team()


def models_improvement():
    kpis([("Baseline GBM", "0.7958", "Train 5-fold CV Macro F1"),
          ("최종 모델", "0.8441", "5-fold CV · 반복 CV 0.8389"),
          ("누적 개선", "+0.0483", "가장 큰 개선: 파생변수 +0.0176")], accent_first=True)
    figure("improvement_steps.png",
           "① 최종 모델까지 단계별 Macro F1 변화(Train 5-fold CV) ② 단계별 최고 모델의 5-fold CV · 반복 CV · Test. "
           "②의 Test 막대는 단계마다 확정된 모델을 확인한 값이며 선정에는 쓰지 않았습니다.")
    st.subheader("개선 경로 (Train 5-fold CV)")
    table(IMPROVEMENT_PATH, {"Macro F1 (5-fold CV)": "%.4f", "직전 대비": "%+.4f"})


def models_imbalance():
    figure("imbalance_comparison.png",
           "① GBM 기본값에 불균형 처리 방법만 바꿨을 때의 Macro F1(반복 CV 15번) ② 처리 없음 대비 유형별 Recall 변화")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("방법별 Macro F1")
        df = IMBALANCE_CV.copy()
        df["처리 없음 대비"] = df["Macro F1 (반복 CV 평균)"] - df.loc[0, "Macro F1 (반복 CV 평균)"]
        table(df, {"Macro F1 (반복 CV 평균)": "%.3f", "처리 없음 대비": "%+.3f"})
    with right:
        st.subheader("모델별 처리 전후 (5-fold CV)")
        cv = CV_TABLE.set_index("모델")
        pairs = [("Random Forest", "Random Forest + SMOTENC"), ("GBM", "GBM + 클래스 가중치"),
                 ("XGBoost", "XGBoost + SMOTENC"), ("LightGBM", "LightGBM + SMOTENC")]
        df = pd.DataFrame([{"모델": before, "처리 방법": cv.loc[after, "불균형 처리"],
                            "처리 전": cv.loc[before, "Macro F1 (5-fold)"], "처리 후": cv.loc[after, "Macro F1 (5-fold)"]}
                           for before, after in pairs])
        df["변화"] = df["처리 후"] - df["처리 전"]
        table(df, {"처리 전": "%.4f", "처리 후": "%.4f", "변화": "%+.4f"})
    note("가중치·과대표집 방법은 +0.01 남짓 개선되지만 서로의 차이는 fold 간 흔들림 범위 안입니다. "
         "Pastry·Bumps Recall이 오르는 만큼 Other_Faults Recall이 내려가는 맞바꾸기가 있고, 언더샘플링은 정보 손실로 크게 떨어졌습니다.")


def models_candidates():
    st.subheader("후보 모델 비교 — Train 내부 CV, 같은 fold · 같은 지표")
    note("모든 값은 Train 1,552건 안에서 팀 공통 5-fold CV(StratifiedKFold, shuffle, random_state=42)로 계산했습니다.")
    left, right = st.columns([1.3, 1])
    with left:
        metric = st.radio("지표", ["Macro F1 (5-fold)", "Macro F1 (반복 CV 5×3)", "Weighted F1", "Accuracy",
                                   "Macro Precision", "Macro Recall", "Macro ROC-AUC"], horizontal=True)
    with right:
        stages = st.multiselect("단계", STAGES, default=STAGES)

    df = CV_TABLE[CV_TABLE["단계"].isin(stages)].dropna(subset=[metric]).sort_values(metric, ascending=False)
    if df.empty:
        st.info("선택한 단계에는 이 지표가 기록되어 있지 않습니다.")
    else:
        df["구분"] = np.where(df["모델"] == FINAL_CV_NAME, "최종 모델", df["단계"])
        lo = max(0.0, np.floor((df[metric].min() - 0.02) * 50) / 50)      # x축 시작값
        hi = min(1.0, df[metric].max() + 0.015)
        palette = {"① Baseline": GRAY, "② 불균형 처리": "#c6d8ee", "③ 튜닝": BLUE_LIGHT, "④ 파생변수": "#5d97dc",
                   "⑤ 재모델링·앙상블": "#3f7fc9", "최종 모델": "#0d366b"}
        shown = [s for s in STAGES if s in set(df["구분"])] + (["최종 모델"] if (df["구분"] == "최종 모델").any() else [])
        color = alt.Color("구분:N", scale=alt.Scale(domain=shown, range=[palette[s] for s in shown]), title=None)
        c = hbar(df, "모델", metric, sort=list(df["모델"]), color=color, x_title=metric, domain=(lo, hi), start=lo, bar=17,
                 y_axis=alt.Axis(labelOverlap=False),
                 tooltip=["모델", "단계", "피처", "불균형 처리", alt.Tooltip(f"{metric}:Q", format=".4f"), "노트북"])
        chart(c.properties(height=max(200, 28 * len(df))))
        caption(f"값이 높은 순서로 정렬했습니다. x축은 {lo:.2f}부터 시작합니다. 표시 {len(df)}개 모델.")
    number_cols = {c: "%.4f" for c in CV_TABLE.columns if CV_TABLE[c].dtype.kind == "f"}
    table(CV_TABLE[CV_TABLE["단계"].isin(stages)], number_cols, height=420)


def models_ranking():
    st.subheader("최종 모델 선정 순위 — 반복 층화 CV(5×3) 평균")
    score = "Macro F1 (반복 CV 5×3)"
    rank = CV_TABLE.dropna(subset=[score]).sort_values(score, ascending=False)
    rank.insert(0, "순위", range(1, len(rank) + 1))
    rank["직전 최고보다 좋은 fold"] = rank["모델"].map(BETTER_FOLDS).fillna("—")
    rank["선정"] = np.where(rank["모델"] == FINAL_CV_NAME, "✅ 최종", "")
    rank["1위와 차이"] = rank[score] - rank[score].max()

    color = alt.condition(alt.datum["모델"] == FINAL_CV_NAME, alt.value("#0d366b"), alt.value(BLUE_LIGHT))
    c = hbar(rank, "모델", score, sort=list(rank["모델"]), color=color, x_title="반복 CV(5×3) Macro F1",
             domain=(0.79, 0.845), start=0.79,
             tooltip=["순위", "모델", alt.Tooltip(f"{score}:Q", format=".4f"), alt.Tooltip("1위와 차이:Q", format="+.4f")])
    goal = alt.Chart(pd.DataFrame({"x": [0.83]})).mark_rule(color=RED, strokeDash=[4, 3]).encode(x="x:Q")
    chart((c + goal).properties(height=max(220, 28 * len(rank))))
    caption("빨간 점선 = 목표 0.83. x축은 0.79부터 시작합니다.")
    table(rank[["순위", "선정", "모델", "단계", score, "Macro F1 (5-fold)", "직전 최고보다 좋은 fold"]],
          {"순위": "%d", score: "%.4f", "Macro F1 (5-fold)": "%.4f"})
    note("<b>선정 규칙(Test를 보기 전에 고정)</b>: 반복 CV 평균이 직전 최고 모델보다 높고 15개 fold 중 과반에서 좋으면 교체. "
         "1위 Soft Voting(ET+GBM)과 2위 최종 모델(ET+RF+GBM)의 차이는 0.0002로, 실행 환경에 따라 순위가 바뀌는 수준입니다. "
         "06 노트북 첫 실행에서 규칙에 따라 ET+RF+GBM이 먼저 확정되었고, Test 재사용을 막기 위해 이후 다시 고르지 않았습니다.")


def models_team():
    scores = load_scores()
    if scores is None:
        st.warning("results/model_scores.csv 파일을 찾지 못했습니다.")
        return
    view = st.radio("보기", ["같은 기준 (팀 전체)", "팀원별 기준"], horizontal=True, label_visibility="collapsed")
    if view == "같은 기준 (팀 전체)":
        team_same_basis(scores)
    else:
        team_by_member(scores)


def team_same_basis(scores: pd.DataFrame):
    note("팀원 전체를 <b>한 가지 기준</b>으로 비교합니다: 팀 공통 config.CV(5-fold)에서 기록한 <b>Macro F1</b>. "
         "세 사람의 같은 설정(GradientBoosting 기본값) 점수가 0.794~0.796으로 거의 같아 같은 기준임을 확인할 수 있습니다. "
         "Weighted F1·정확도는 일부 실험에 기록이 없어 이 비교에서는 제외했습니다.")
    df = scores.assign(실험=scores["실험명"] + " (" + scores["담당자"] + ")").sort_values("Macro_F1", ascending=False)
    people = list(dict.fromkeys(scores["담당자"]))
    lo = np.floor((df["Macro_F1"].min() - 0.01) * 100) / 100
    color = alt.Color("담당자:N", scale=alt.Scale(domain=people, range=[BLUE, ORANGE, GREEN, RED, GRAY][:len(people)]), title=None)
    c = hbar(df, "실험", "Macro_F1", sort=list(df["실험"]), color=color, x_title="Macro F1 (팀 공통 5-fold CV)",
             domain=(lo, df["Macro_F1"].max() + 0.01), start=lo, text_size=11, y_axis=alt.Axis(labelOverlap=False),
             tooltip=["담당자", "실험명", alt.Tooltip("Macro_F1:Q", format=".4f"), "비고"])
    chart(c.properties(height=max(240, 30 * len(df))))
    caption(f"높은 순서로 정렬, x축은 {lo:.2f}부터 시작합니다.")

    st.subheader("담당자별 최고 기록")
    best = df.loc[df.groupby("담당자")["Macro_F1"].idxmax()].sort_values("Macro_F1", ascending=False)
    baseline = scores[scores["실험명"].str.contains("baseline", case=False)].set_index("담당자")["Macro_F1"]
    best["Baseline"] = best["담당자"].map(baseline)
    best["Baseline 대비"] = best["Macro_F1"] - best["Baseline"]
    table(best[["담당자", "실험명", "Baseline", "Macro_F1", "Baseline 대비", "비고"]],
          {"Baseline": "%.4f", "Macro_F1": "%.4f", "Baseline 대비": "%+.4f"})


def team_by_member(scores: pd.DataFrame):
    note("담당자마다 기록한 지표(Macro F1 · Weighted F1 · 정확도)를 실험 순서대로 보여줍니다. 비어 있는 값은 기록하지 않은 지표입니다.")
    metrics = ["Macro_F1", "Weighted_F1", "정확도"]
    people = list(dict.fromkeys(scores["담당자"]))
    for who, tab in zip(people, st.tabs(people)):
        with tab:
            g = scores[scores["담당자"] == who].copy()
            g["실험"] = g["순서"].astype(str) + ". " + g["실험명"]
            long = g.melt(id_vars=["실험", "실험명"], value_vars=metrics, var_name="지표", value_name="값").dropna(subset=["값"])
            lo = np.floor((long["값"].min() - 0.01) * 100) / 100
            color = alt.Color("지표:N", scale=alt.Scale(domain=metrics, range=[BLUE, BLUE_LIGHT, GRAY]), title=None)
            c = hbar(long, "실험", "값", sort=list(g["실험"]), color=color, x_title="CV 점수",
                     domain=(lo, long["값"].max() + 0.012), start=lo, bar=11, rounded=False, text_size=10, dx=4,
                     group="지표", group_sort=metrics, tooltip=["실험명", "지표", alt.Tooltip("값:Q", format=".4f")])
            chart(c.properties(height=max(200, 84 * len(g))))
            table(g[["순서", "실험명", "Macro_F1", "Weighted_F1", "정확도", "비고"]],
                  {"순서": "%d", "Macro_F1": "%.4f", "Weighted_F1": "%.4f", "정확도": "%.4f"})


# ══════════════════════════════════════════════════════════════════════════════
# 4. 최종 모델 평가 (Test 는 최종 모델 1개에만 사용)
# ══════════════════════════════════════════════════════════════════════════════
def page_final():
    page_header("최종 모델 평가", "CV로 선정한 Soft Voting(ExtraTrees + Random Forest + GBM, 피처 36개)을 Train 1,552건 전체로 학습하고 "
                               "Test 389건으로 한 번 평가한 결과입니다. Test는 이 모델 하나에만 사용했습니다.")
    F = FINAL
    kpis([("Test Macro F1", f"{F['macro_f1']:.3f}", f"95% CI {F['ci'][0]:.3f} ~ {F['ci'][1]:.3f}"),
          ("반복 CV(5×3) — 선정 기준", f"{F['rep_cv']:.3f}", f"표준편차 {F['rep_cv_std']:.3f} · 5-fold {F['cv5']:.3f}"),
          ("Test Accuracy", f"{F['accuracy']:.3f}", f"오분류 {int(FINAL_CM.sum() - np.trace(FINAL_CM))}건 / 389건"),
          ("Test Weighted F1", f"{F['weighted_f1']:.3f}", f"Macro ROC-AUC {F['roc_auc']:.3f}"),
          ("Precision / Recall", f"{F['precision']:.3f} / {F['recall']:.3f}", "Macro 평균")],
         accent_first=True)
    note(f"Test는 1회 분할이라 변동이 큽니다(신뢰구간 폭 약 ±0.04). 부트스트랩 기준 목표 0.83 이상일 확률은 {F['p_ge_083']:.0%}이며, "
         "CV 추정(0.839)보다 0.007 낮지만 신뢰구간 안입니다.")

    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.subheader("Confusion Matrix")
        mode = st.radio("값", ["행 기준 %", "건수"], horizontal=True, label_visibility="collapsed")
        confusion_chart(FINAL_CM, f"최종 모델 Test 혼동행렬 (Macro F1 {F['macro_f1']:.3f})", mode)
        caption("행 = 실제 유형, 열 = 예측 유형. 칸의 큰 숫자는 그 행에서의 비율, 괄호는 건수입니다. "
                "오른쪽 색 막대는 칸 색이 나타내는 값의 범위입니다.")
    with right:
        st.subheader("유형별 Precision · Recall · F1")
        metrics = ["Precision", "Recall", "F1"]
        long = FINAL_PER_CLASS.reset_index(names="유형").melt(id_vars="유형", var_name="지표", value_name="값")
        color = alt.Color("지표:N", scale=alt.Scale(domain=metrics, range=[BLUE_LIGHT, GRAY, BLUE]), title=None)
        c = hbar(long, "유형", "값", sort=CLASS_ORDER, color=color, x_title="Test 점수", domain=(0.4, 1.0), start=0.4,
                 bar=10, rounded=False, fmt=".3f", text_size=10, dx=4, group="지표", group_sort=metrics,
                 tooltip=["유형", "지표", alt.Tooltip("값:Q", format=".3f")])
        chart(c.properties(height=460))

    st.markdown(
        "- **잘 맞히는 유형**: Z_Scratch(F1 0.973), K_Scratch(0.954), Dirtiness(0.952), Stains(0.889).\n"
        "- **가장 약한 유형은 Pastry**(F1 0.545)로, Test 32건 중 18건만 맞혔습니다.\n"
        "- **최대 오류원은 Bumps ↔ Other_Faults**(32건, 전체 오분류 74건의 43%), 다음은 Other_Faults ↔ Pastry(21건)입니다.\n"
        "- 샘플러 없이 클래스 가중치만 써서 Precision(0.849)이 Recall(0.818)보다 높습니다.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("유형별 상세 지표")
        detail = FINAL_PER_CLASS.copy()
        detail.insert(0, "Test 건수", [TEST_N[c] for c in CLASS_ORDER])
        table(detail.reset_index(names="유형"), {"Test 건수": "%d", "Precision": "%.3f", "Recall": "%.3f", "F1": "%.3f"})
        st.subheader("주요 오류 건수")
        table(FINAL_KEY_ERRORS, {"건수": "%d"})
    with right:
        st.subheader("피처 중요도 (Permutation, Test)")
        importance_chart()

    with st.expander("저장된 모델로 Test 결과 재검증 (재학습 없이 예측만 수행)"):
        st.write("대시보드에 표시한 수치가 실제 실험 결과와 같은지, 저장된 모델로 Test 389건을 다시 예측해 비교합니다.")
        if st.button("Test 데이터로 재검증", type="primary"):
            verify_final_model()


def importance_chart():
    """Permutation 중요도 상위 15개 (막대 = 평균, 가는 선 = ± 표준편차)."""
    imp = PERM_IMPORTANCE.copy()
    imp["구분"] = np.where(imp["파생변수"], "새로 만든 파생변수", "원본 변수")
    imp["lo"], imp["hi"] = imp["F1 감소량"] - imp["표준편차"], imp["F1 감소량"] + imp["표준편차"]
    order = list(imp["피처"])
    bars = alt.Chart(imp).mark_bar(height=14).encode(
        y=alt.Y("피처:N", sort=order, title=None), x=alt.X("F1 감소량:Q", title="섞었을 때 Test Macro F1 감소량"),
        color=alt.Color("구분:N", scale=alt.Scale(domain=["원본 변수", "새로 만든 파생변수"], range=[BLUE, RED]), title=None),
        tooltip=["피처", alt.Tooltip("F1 감소량:Q", format=".4f"), alt.Tooltip("표준편차:Q", format=".4f")])
    errors = alt.Chart(imp).mark_rule(color=MUTED).encode(y=alt.Y("피처:N", sort=order), x="lo:Q", x2="hi:Q")
    chart((bars + errors).properties(height=430))


def verify_final_model():
    """저장된 모델로 Test 를 다시 예측해 대시보드 수치(노트북 결과)와 같은지 확인합니다."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    model, model_desc, error = load_model()
    if model is None:
        show_error("저장된 모델을 불러오지 못했습니다.", error)
        return
    split, data_desc, error = load_split()
    if split is None:
        show_error("Train/Test 데이터를 불러오지 못했습니다.", error)
        return
    X_train, X_test, _, y_test = split
    try:
        with st.spinner("Test 389건 예측 중…"):
            proba = predict_proba(model, X_test, X_train)
            pred = proba.columns.values[proba.values.argmax(axis=1)].astype(str)
    except Exception:
        show_error("예측 중 오류가 났습니다.", traceback.format_exc())
        return

    live = {"Test Macro F1": f1_score(y_test, pred, average="macro"),
            "Test Weighted F1": f1_score(y_test, pred, average="weighted"),
            "Test Accuracy": accuracy_score(y_test, pred)}
    saved = {"Test Macro F1": FINAL["macro_f1"], "Test Weighted F1": FINAL["weighted_f1"], "Test Accuracy": FINAL["accuracy"]}
    result = pd.DataFrame({"지표": list(live), "대시보드 값": [saved[k] for k in live], "재계산 값": list(live.values())})
    result["차이"] = result["재계산 값"] - result["대시보드 값"]
    cm_live = confusion_matrix(y_test, pred, labels=CLASS_ORDER)

    st.caption(f"모델: {model_desc} · 데이터: {data_desc} · 파생변수: src/Taegyeom_model_utils.add_features")
    table(result, {"대시보드 값": "%.4f", "재계산 값": "%.4f", "차이": "%+.4f"})
    if np.all(np.abs(result["차이"]) < 5e-4) and np.array_equal(cm_live, FINAL_CM):
        st.success("지표와 혼동행렬이 모두 노트북 결과와 일치합니다.")
    else:
        st.warning("일부 값이 다릅니다. 모델 파일이 최종 노트북에서 저장한 것인지, 라이브러리 버전이 같은지 확인해 주세요.")
        confusion_chart(cm_live, "재계산한 혼동행렬", "건수")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 일반화 점검
# ══════════════════════════════════════════════════════════════════════════════
def page_generalization():
    page_header("일반화 점검", "일반 CV는 같은 생산 조건의 결함이 Train과 검증에 함께 들어가 점수가 높게 나옵니다. "
                            "생산 조건 묶음을 통째로 검증에 빼는 그룹 CV로 '처음 보는 조건'에서의 성능을 따로 확인했습니다.")
    kpis([("최종 모델 일반 CV", "0.840", "이미 본 생산 조건"),
          ("최종 모델 그룹 CV", "0.740", "처음 보는 생산 조건 · −0.100"),
          ("plate_defect_count 제외 시 그룹 CV", "0.780", "일반 CV는 0.840으로 동일")], accent_first=True)
    figure("generalization.png",
           "① 모델별 일반 CV(연한 막대)와 그룹 CV(진한 막대), 반복 15번 평균 ± 표준편차 ② 같은 fold끼리 비교한 점수 차이")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("확인한 것")
        st.markdown(
            "- **모든 모델이 처음 보는 조건에서 크게 떨어집니다.** 공정 조건 3개만 쓴 모델은 0.672 → 0.559, Baseline XGBoost는 0.814 → 0.728입니다.\n"
            "- **파생변수의 효과는 두 기준에서 정반대입니다.** 파생변수를 빼면 일반 CV는 0.013 낮아지지만 그룹 CV는 0.030 높아집니다.\n"
            "- 특히 강판별 결함 수(`plate_defect_count`)는 이미 본 강판에서는 도움이 되지만, 새 강판에서는 오히려 방해가 됩니다(제외 시 그룹 CV +0.040, 15번 중 11번 우세).")
    with right:
        st.subheader("활용 시 주의사항")
        st.markdown(
            "- 대시보드의 Test 점수(0.832)는 **이미 학습한 생산 조건이 대부분인 상황**의 성능입니다. 새 강종·새 폭·새 두께에서는 0.74 수준을 예상해야 합니다.\n"
            "- Test에서도 Train에 없던 조건의 결함은 오분류율이 29.2%로, 이미 본 조건(18.4%)보다 높았습니다.\n"
            "- 새 생산 조건이 많은 라인에 적용한다면 `plate_defect_count`를 뺀 모델을 쓰는 것이 낫습니다.\n"
            "- 예측 체험 화면은 처음 보는 생산 조건이면 **검사자 재확인 필요**로 표시합니다.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 예측 체험
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_GROUPS = {    # 입력 칸을 묶어 보여줄 순서
    "공정 조건": ["Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300", "TypeOfSteel_A400"],
    "위치": ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum"],
    "크기·형상": ["Pixels_Areas", "X_Perimeter", "Y_Perimeter", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "SigmoidOfAreas"],
    "밝기": ["Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Luminosity_Index"],
    "형태 지수": ["Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
               "Outside_Global_Index", "Orientation_Index"],
}
INTEGER_FEATURES = {"X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
                    "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
                    "Steel_Plate_Thickness", "TypeOfSteel_A300"}


def page_predict():
    page_header("예측 체험", "결함 측정값을 넣으면 최종 모델이 유형별 확률을 계산합니다. 확신이 낮거나 처음 보는 생산 조건이면 "
                          "검사자가 다시 확인하도록 표시합니다.")
    model, model_desc, error = load_model()
    if model is None:
        show_error("저장된 최종 모델을 불러오지 못했습니다.", error)
        return
    split, _, error = load_split()
    if split is None:
        show_error("입력 시작값으로 쓸 Train/Test 데이터를 불러오지 못했습니다.", error)
        return
    X_train, X_test, _, y_test = split
    features = list(X_train.columns)

    st.caption(f"사용 모델: {model_desc}")
    if len(model.members) > 1:
        st.info("저장 파일에 모델이 여러 개 들어 있어 구성 모델의 예측 확률을 평균해 씁니다(Soft Voting과 같은 방식). "
                "최종 모델 평가 화면의 '재검증'으로 노트북 결과와 같은지 확인해 보세요.")

    # 재확인 기준 (사이드바)
    with st.sidebar:
        st.markdown("### 재확인 기준")
        rules = dict(
            thr=st.slider("최고 확률이 이 값보다 낮으면", 0.30, 0.90, 0.60, 0.05),
            gap=st.slider("1·2순위 확률 차이가 이 값보다 작으면", 0.00, 0.50, 0.15, 0.05),
            use_plate=st.checkbox("생산 조건도 확인", value=True,
                                  help="Train에 같은 생산 조건(컨베이어 길이·두께·강종)의 결함이 적으면 재확인으로 표시합니다."))
        rules["min_same"] = st.slider("Train에 같은 조건 결함이 이 건수보다 적으면", 1, 10, 1, 1, disabled=not rules["use_plate"],
                                      help="1이면 처음 보는 조건(0건)일 때만 표시합니다.")

    # 입력 시작값: Test 샘플 또는 Train 중앙값
    c1, c2, c3 = st.columns([1.1, 1, 1.4], gap="medium")
    with c1:
        source = st.radio("시작값", ["Test 샘플", "Train 중앙값"], horizontal=True)
    sample_id, true_label = "median", None
    if source == "Test 샘플":
        with c2:
            cls = st.selectbox("실제 유형으로 거르기", ["전체"] + CLASS_ORDER)
        ids = list(X_test.index if cls == "전체" else y_test[y_test == cls].index)
        with c3:
            sample_id = st.selectbox("Test 샘플 번호", ids, format_func=lambda i: f"{i}  (실제: {y_test.loc[i]})")
        start, true_label = X_test.loc[sample_id], y_test.loc[sample_id]
    else:
        start = X_train.median(numeric_only=True)

    result_area = st.container()          # 결과는 입력 폼 위에 표시
    values, submitted = input_form(features, start, sample_id, X_train)

    if submitted:
        row = pd.DataFrame([values], columns=features)
        if "TypeOfSteel_A400" in row and "TypeOfSteel_A300" in row:
            row["TypeOfSteel_A400"] = 1 - row["TypeOfSteel_A300"]
        try:
            proba = predict_proba(model, row.astype(X_train.dtypes.to_dict()), X_train).iloc[0]
        except Exception:
            with result_area:
                show_error("예측 중 오류가 났습니다.", traceback.format_exc())
            return
        plate = tuple(float(values[k]) for k in PLATE_KEYS)
        st.session_state["last_pred"] = dict(proba=proba, plate=plate, true=true_label, sample=sample_id)

    with result_area:
        last = st.session_state.get("last_pred")
        if last is None or last["sample"] != sample_id:
            st.info("아래 측정값을 확인하고 '유형 예측하기'를 누르면 결과가 여기에 표시됩니다.")
            return
        show_prediction(last, rules, plate_counts(X_train))


def input_form(features, start, sample_id, X_train):
    """측정값 입력 폼. 반환: (입력값 dict, 제출 여부)"""
    st.subheader("측정값")
    values = {}
    with st.form("predict_form"):
        for group, names in FEATURE_GROUPS.items():
            names = [f for f in names if f in features]
            if not names:
                continue
            st.markdown(f"**{group}**")
            cols = st.columns(4)
            for k, feat in enumerate(names):
                v = float(start.get(feat, 0.0))
                key = f"in_{feat}_{sample_id}"          # 시작값을 바꾸면 입력 칸도 새 값으로
                with cols[k % 4]:
                    if feat.startswith("TypeOfSteel"):
                        values[feat] = st.selectbox(feat, [0, 1], index=int(round(v)), key=key)
                    elif pd.api.types.is_integer_dtype(X_train[feat]) or feat in INTEGER_FEATURES:
                        values[feat] = st.number_input(feat, value=int(round(v)), step=1, key=key)
                    else:
                        values[feat] = st.number_input(feat, value=v, format="%.4f", key=key)
        submitted = st.form_submit_button("유형 예측하기", type="primary")
    return values, submitted


def plate_counts(X_train) -> dict:
    """생산 조건(컨베이어 길이, 두께, 강종)별 Train 결함 수."""
    return pd.Series(list(map(tuple, X_train[PLATE_KEYS].astype(float).values.tolist()))).value_counts().to_dict()


def review_checks(proba: pd.Series, plate: tuple, rules: dict, counts: dict) -> list[dict]:
    """재확인 기준을 하나씩 판정합니다. 반환: [{기준, 판정 근거, 결과}, ...]"""
    top = proba.sort_values(ascending=False)
    c1, c2 = top.index[0], top.index[1]
    p1, p2 = float(top.iloc[0]), float(top.iloc[1])
    thr, gap, min_same = rules["thr"], rules["gap"], rules["min_same"]
    checks = [{"기준": "① 최고 확률", "판정 근거": f"{c1} {p1:.1%}  (기준 {thr:.0%} 이상)",
               "결과": "⚠️ 해당" if p1 < thr else "✅ 통과"},
              {"기준": "② 1·2순위 확률 차이",
               "판정 근거": f"1위 {c1} {p1:.1%} − 2위 {c2} {p2:.1%} = {(p1 - p2) * 100:.1f}%p  (기준 {gap * 100:.0f}%p 이상)",
               "결과": "⚠️ 해당" if p1 - p2 < gap else "✅ 통과"}]
    if rules["use_plate"]:
        n_same = int(counts.get(plate, 0))
        condition = f"컨베이어 길이 {plate[0]:g} · 두께 {plate[1]:g} · 강종 {'A300' if plate[2] == 1 else 'A400'}"
        checks.append({"기준": "③ 생산 조건",
                       "판정 근거": f"{condition} → Train 에 같은 조건 결함 {n_same}건  (기준 {min_same}건 이상)"
                                    + ("  · 처음 보는 조건, Test 오분류율 29.2%" if n_same == 0 else ""),
                       "결과": "⚠️ 해당" if n_same < min_same else "✅ 통과"})
    return checks


def show_prediction(last: dict, rules: dict, counts: dict):
    proba, true_label = last["proba"], last["true"]
    top = proba.sort_values(ascending=False)
    c1, c2 = top.index[0], top.index[1]
    p1, p2 = float(top.iloc[0]), float(top.iloc[1])
    thr, gap = rules["thr"], rules["gap"]
    checks = review_checks(proba, last["plate"], rules, counts)
    flagged = [c["기준"][2:] for c in checks if c["결과"].startswith("⚠️")]

    st.header("예측 결과")
    if flagged:
        st.markdown(f'<div class="verdict check"><b>검사자 재확인 필요</b> &nbsp; 예측: {c1} ({p1:.1%})'
                    f'<br>해당 기준: {" · ".join(flagged)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict ok"><b>자동 판정 가능</b> &nbsp; 예측: {c1} ({p1:.1%})'
                    f'<br>재확인 기준을 모두 통과했습니다.</div>', unsafe_allow_html=True)
    table(pd.DataFrame(checks))
    if true_label is not None:
        st.caption(f"이 Test 샘플의 실제 유형은 **{true_label}** 입니다 "
                   f"({'맞힘' if true_label == c1 else '틀림'}, 입력값을 고치지 않은 경우 기준).")

    df = pd.DataFrame({"유형": proba.index, "확률": proba.values.astype(float)})
    df["구분"] = np.where(df["유형"] == c1, "예측 유형", "기타")
    color = alt.Color("구분:N", scale=alt.Scale(domain=["예측 유형", "기타"], range=[BLUE, BLUE_LIGHT]), legend=None)
    c = hbar(df, "유형", "확률", sort=CLASS_ORDER, color=color, x_title="예측 확률", domain=(0, 1), bar=22, fmt=".1%",
             x_axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
             tooltip=["유형", alt.Tooltip("확률:Q", format=".1%")])
    threshold = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(strokeDash=[4, 3], color=ORANGE).encode(x="x:Q")
    chart((c + threshold).properties(height=300))
    caption(f"주황 점선 = 재확인 기준 확률 {thr:.2f}. 기준은 왼쪽 사이드바에서 바꿀 수 있습니다.")
    if c1 in CONFUSED and c2 in CONFUSED and p1 - p2 < max(gap, 0.30):
        note(f"1·2순위({c1}, {c2})가 모두 서로 헷갈리는 유형이고 확률 차이도 크지 않습니다. "
             "Test 오분류의 대부분이 Other_Faults·Bumps·Pastry 사이에서 나왔습니다.")


# ══════════════════════════════════════════════════════════════════════════════
# 화면 선택
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "데이터": page_data,
    "데이터 구조": page_structure,
    "모델 비교": page_models,
    "최종 모델 평가": page_final,
    "일반화 점검": page_generalization,
    "예측 체험": page_predict,
}

with st.sidebar:
    st.markdown("## 강판 결함 분류")
    st.caption("Steel Plates Faults · 7개 유형 · 최종 모델 Soft Voting")
    page = st.radio("화면", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()

PAGES[page]()
