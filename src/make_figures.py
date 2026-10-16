"""README 그림 15장 재생성 스크립트.

README에 실린 figures/*.png 를 만든 코드를 그대로 모은 것이다.

기본 동작은 README 그림을 **덮어쓰지 않는다.**
  uv run python src/make_figures.py                       # figures/regenerated/ 에 저장
  uv run python src/make_figures.py --only overview       # 일부만 (쉼표로 여러 개)
  uv run python src/make_figures.py --list                # 그림 이름 목록
  uv run python src/make_figures.py --overwrite           # figures/ 에 직접 저장
                                                          #  (기존 그림은 figures/_backup_날짜시간/ 으로 먼저 복사)

데이터 출처
- data/processed/Faults_processed.csv, data/split/*.csv : 팀 공통 데이터와 분할
- results/figure_data/ : 시간이 오래 걸리는 실험 결과를 저장해 둔 파일
    cv_strat_vs_group.json         05_team_verification.ipynb 6절 (일반 CV vs 그룹 CV, 15 fold별 점수)
    imbalance_repeated_cv.json     05_team_verification.ipynb 7절 (불균형 처리 반복 비교)
    final_model_oof_5fold.csv      07_final_model.ipynb 4-1 셀과 같은 5-fold의 검증 예측
    permutation_importance_test.csv 07_final_model.ipynb 6-2 셀 (원본 26개 열 순열 중요도)
- 그 밖의 값(Test 혼동행렬, 학습곡선, 단계별 개선 점수 등)은 07_final_model.ipynb와
  Taegyeom_02~07 노트북의 공식 출력값을 코드 안에 그대로 적었다.

글꼴이나 라이브러리 버전에 따라 그림 모양이 README와 조금 다를 수 있다.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.ticker as mt
from matplotlib import font_manager as fm

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "src" / "config.py").exists())
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "results" / "figure_data"
RS = 42

# ───────────────────────── 공통 스타일 ─────────────────────────
def _set_font():
    for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ["Noto Sans CJK JP", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if name in have:
            return name
    print("[경고] 한글 글꼴을 찾지 못했습니다. 글자가 깨질 수 있습니다.")
    return "DejaVu Sans"


plt.rcParams.update({"font.family": _set_font(), "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12.5, "figure.dpi": 110, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": False})

KO = {k: k for k in ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]}
KO1 = dict(KO)
EASY, HARD, GREY, DARK, GOOD, BAD = "#2E6FBA", "#E07B29", "#9AA3AD", "#2B2B2B", "#3A9D5D", "#C8453B"
HARDSET = {"Bumps", "Pastry", "Other_Faults"}


def ccol(c):
    return HARD if c in HARDSET else EASY


OUT_DIR = FIG_DIR / "regenerated"   # main()에서 결정


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("  저장:", (OUT_DIR / f"{name}.png").relative_to(ROOT))


# ───────────────────────── 데이터 ─────────────────────────
def load():
    df = pd.read_csv(ROOT / "data/processed/Faults_processed.csv")
    tr = pd.read_csv(ROOT / "data/split/train_idx.csv")["index"].to_numpy()
    te = pd.read_csv(ROOT / "data/split/test_idx.csv")["index"].to_numpy()
    return df, tr, te


def cached(name):
    p = DATA_DIR / name
    if not p.exists():
        sys.exit(f"[오류] {p.relative_to(ROOT)} 가 없습니다. 05/07 노트북 결과로 다시 만들어야 합니다.")
    return json.load(open(p, encoding="utf-8")) if p.suffix == ".json" else pd.read_csv(p)


# ───────────────────────── 그림 ─────────────────────────
def overview():
    fig, ax = plt.subplots(figsize=(14, 4.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=12, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 0.75, body, ha="center", va="top", fontsize=9.6, color="#222", linespacing=1.55)

    steps = [("① 데이터", "UCI 강판 결함 1,941건\n측정값 27개 · 결함 7종\nTrain 1,552 / Test 389"),
             ("② 점검·전처리", "정답 열 통합 · 오타 수정\n중복 강종 변수 제거\n유사 중복 · 극단값 점검"),
             ("③ 탐색(EDA)", "변수별 분리력\n공정 조건 묶음 구조\n롤 마크 주기 결함"),
             ("④ 모델링", "Baseline 6종 비교\n불균형 처리 · 튜닝\n파생변수 · 앙상블"),
             ("⑤ 검증", "반복 교차검증\n그룹 교차검증\n과적합 · Test 평가")]
    w, gap = 2.45, 0.33
    for i, (t, b) in enumerate(steps):
        x = 0.15 + i * (w + gap); box(x, 2.75, w, 1.75, t, b, "#EEF4FB", EASY)
        if i < 4:
            ax.annotate("", xy=(x + w + gap - 0.04, 3.62), xytext=(x + w + 0.04, 3.62),
                        arrowprops=dict(arrowstyle="-|>", color=EASY, lw=2))
    ax.text(7, 2.42, "최종 모델: ExtraTrees + RandomForest + GBM 소프트보팅 (클래스 가중치, 변수 36개)",
            ha="center", fontsize=11.5, fontweight="bold", color=DARK)
    # README 1.3절 '보고서의 핵심 주장' 표와 같은 내용
    claims = [("주장 ①", "어려움의 정체는\n\"적은 유형\"이 아니라\nBumps·Pastry·\nOther_Faults의 경계", HARD),
              ("주장 ②", "Baseline보다 개선\nCV 0.811 → 0.844\n반복 교차검증에서도 유지", GOOD),
              ("주장 ③", "강종·두께·폭(생산 조건)에\n크게 의존. 처음 보는 조건에선\n성능 하락, 모델 간 우열도 흐려짐", BAD),
              ("주장 ④", "같은 생산 조건에서\n유사한 결함이 반복\n→ 일반 CV는 '이미 본 조건' 기준", BAD)]
    w2 = 3.25
    for i, (t, b, c) in enumerate(claims):
        box(0.15 + i * (w2 + 0.2), 0.05, w2, 2.0, t, b, "white", c)
    fig.tight_layout(); save(fig, "overview")


def target_distribution():
    df, tr, te = load()
    order = ["Other_Faults", "Bumps", "K_Scratch", "Z_Scratch", "Pastry", "Stains", "Dirtiness"]
    ctr = df.loc[tr].target.value_counts()[order]; cte = df.loc[te].target.value_counts()[order]
    # Baseline XGBoost 유형별 F1 (Train 5-fold CV, 03_modeling.ipynb 출력값)
    f1 = {"K_Scratch": 0.968, "Z_Scratch": 0.950, "Stains": 0.939, "Dirtiness": 0.854,
          "Other_Faults": 0.752, "Bumps": 0.676, "Pastry": 0.561}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    y = np.arange(len(order))[::-1]
    ax[0].barh(y, ctr.values, color=[ccol(c) for c in order], label="Train")
    ax[0].barh(y, cte.values, left=ctr.values, color=[ccol(c) for c in order], alpha=.4, label="Test")
    for yi, c in zip(y, order):
        tot = ctr[c] + cte[c]; ax[0].text(tot + 8, yi, f"{tot}건 ({tot / len(df):.1%})", va="center", fontsize=10)
    ax[0].set_yticks(y, [KO1[c] for c in order]); ax[0].set_xlim(0, 800); ax[0].set_xlabel("결함 건수")
    ax[0].set_title("① 결함 유형별 건수 (진한 색 Train, 연한 색 Test)")
    ax[0].text(790, 6.55, f"가장 많은 유형 ÷ 가장 적은 유형 = {673 / 55:.1f}배", ha="right", fontsize=10, color=DARK)
    n = ctr + cte
    for c in order:
        ax[1].scatter(n[c], f1[c], s=140, color=ccol(c), zorder=3, edgecolor="white")
        off = {"Dirtiness": (8, -0.02), "Stains": (8, -0.025), "Pastry": (8, -0.005), "Bumps": (8, 0.0),
               "Other_Faults": (-8, 0.022), "K_Scratch": (-8, 0.018), "Z_Scratch": (8, -0.002)}[c]
        ax[1].annotate(KO1[c], (n[c], f1[c]), xytext=(n[c] + off[0] * 4, f1[c] + off[1]), fontsize=9.5,
                       ha="left" if off[0] > 0 else "right", va="center")
    ax[1].set_xlim(0, 760); ax[1].set_ylim(0.5, 1.02); ax[1].set_xlabel("결함 건수")
    ax[1].set_ylabel("F1 (Baseline XGBoost, Train 5-fold CV)")
    ax[1].set_title("② 건수가 적다고 맞히기 어려운 것은 아니다")
    ax[1].axhspan(0.5, 0.8, color=HARD, alpha=.07); ax[1].text(745, 0.52, "F1 0.8 미만 구간", ha="right", fontsize=9, color=HARD)
    ax[1].legend(handles=[mp.Patch(color=EASY, label="잘 구분되는 유형"), mp.Patch(color=HARD, label="서로 헷갈리는 유형")],
                 loc="lower right", bbox_to_anchor=(1, 0.08), frameon=False, fontsize=9.5)
    fig.tight_layout(); save(fig, "target_distribution")


def eda_result():
    from scipy.stats import kruskal
    df, tr, te = load(); T = df.loc[tr]
    rows = []
    for c in T.columns.drop("target"):
        H, p = kruskal(*[T.loc[T.target == k, c].values for k in T.target.unique()])
        rows.append((c, H, p, (H - 7 + 1) / (len(T) - 7)))
    kw = pd.DataFrame(rows, columns=["f", "H", "p", "eps2"]).sort_values("eps2", ascending=False)
    grpmap = {}
    for f in ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum"]: grpmap[f] = "위치"
    for f in ["Pixels_Areas", "X_Perimeter", "Y_Perimeter", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
              "SigmoidOfAreas", "Sum_of_Luminosity"]: grpmap[f] = "크기"
    for f in ["Minimum_of_Luminosity", "Maximum_of_Luminosity", "Luminosity_Index"]: grpmap[f] = "밝기"
    for f in ["Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
              "Outside_Global_Index", "Orientation_Index"]: grpmap[f] = "모양"
    for f in ["Length_of_Conveyer", "TypeOfSteel_A300", "Steel_Plate_Thickness"]: grpmap[f] = "공정 조건"
    gc = {"크기": "#2E6FBA", "모양": "#6BAED6", "밝기": "#9E9AC8", "위치": GREY, "공정 조건": "#C8453B"}
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={"width_ratios": [1, 1.15]})
    k = kw.iloc[::-1]
    ax[0].barh(k.f, k.eps2, color=[gc[grpmap[f]] for f in k.f])
    ax[0].set_xlabel("분리력 (Kruskal–Wallis 효과크기 ε²)"); ax[0].tick_params(axis="y", labelsize=8.5)
    ax[0].set_title("① 변수 하나만으로 결함 유형을 얼마나 가르나")
    ax[0].legend(handles=[mp.Patch(color=v, label=k_) for k_, v in gc.items()], loc="lower right", frameon=False,
                 fontsize=9.5, title="변수 묶음", title_fontsize=9.5)
    ax[0].text(0.0, -3.3, "모든 변수 p < 0.001 · Train 1,552건 기준", ha="left", fontsize=8.5, color="#555")
    order = ["K_Scratch", "Z_Scratch", "Stains", "Dirtiness", "Bumps", "Pastry", "Other_Faults"]
    data = [T.loc[T.target == c, "LogOfAreas"] for c in order]
    bp = ax[1].boxplot(data, patch_artist=True, widths=.6, medianprops=dict(color="black"),
                       flierprops=dict(markersize=2.5, alpha=.4))
    for p, c in zip(bp["boxes"], order): p.set_facecolor(ccol(c)); p.set_alpha(.75)
    ax[1].set_xticks(range(1, 8), [KO[c] for c in order], fontsize=8.8)
    ax[1].set_ylabel("결함 면적 (로그 값, LogOfAreas)")
    ax[1].set_title("② 면적만으로는 K_Scratch·Stains만 확실히 갈린다")
    fig.tight_layout(); save(fig, "eda_result")


def process_condition():
    df, tr, te = load()
    df["grp"] = df.TypeOfSteel_A300.astype(str) + "_" + df.Steel_Plate_Thickness.astype(str) + "_" + df.Length_of_Conveyer.astype(str)
    T, Te = df.loc[tr], df.loc[te]

    def purity(d):
        n = d.groupby("grp").size(); c = d.groupby("grp").target.agg(lambda s: s.value_counts().iloc[0] / len(s))
        return (c * n).sum() / n.sum()

    real = purity(T); rng = np.random.default_rng(0); sh = []
    for _ in range(200):
        t = T.copy(); t["target"] = rng.permutation(t.target.values); sh.append(purity(t))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]; a.hist(sh, bins=20, color=GREY, alpha=.8, label="정답을 무작위로 섞었을 때 (200회)")
    a.axvline(real, color=BAD, lw=3, label="실제 데이터"); a.set_xlim(0.38, 0.82)
    a.text(real - 0.005, a.get_ylim()[1] * 0.9, f"실제 {real:.3f}", ha="right", color=BAD, fontweight="bold")
    a.text(np.mean(sh) + 0.012, a.get_ylim()[1] * 0.9, f"무작위 평균 {np.mean(sh):.3f}", color="#555")
    a.set_xlabel("같은 공정 조건 묶음 안에서 가장 많은 유형이 차지하는 비율"); a.set_ylabel("횟수")
    a.set_title("① 같은 공정 조건이면 같은 결함 유형이 반복된다"); a.legend(frameon=False, fontsize=9, loc="center right")
    a = ax[0, 1]; order = ["K_Scratch", "Stains", "Dirtiness", "Other_Faults", "Pastry", "Bumps", "Z_Scratch"]
    s = df.groupby("target").TypeOfSteel_A300.mean()[order]
    yy = np.arange(len(order))[::-1]
    a.barh(yy, 1 - s.values, color="#5B8DB8", label="TypeOfSteel_A300 = 0 (A400)")
    a.barh(yy, s.values, left=1 - s.values, color="#D98C5F", label="TypeOfSteel_A300 = 1 (A300)")
    for y_, v in zip(yy, s.values): a.text(1.01, y_, f"A300 {v:.0%}", va="center", fontsize=9)
    a.set_yticks(yy, [KO1[c] for c in order], fontsize=9.5); a.set_xlim(0, 1.18); a.set_xticks([0, .5, 1], ["0%", "50%", "100%"])
    a.set_title("② TypeOfSteel_A300만 알아도 일부 유형은 거의 정해진다")
    a.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.45, -.2), ncol=2)
    a = ax[1, 0]; order2 = ["Stains", "K_Scratch", "Dirtiness", "Z_Scratch", "Pastry", "Other_Faults", "Bumps"]
    yy = np.arange(len(order2))[::-1]
    for y_, c in zip(yy, order2):
        v = df.loc[df.target == c, "Length_of_Conveyer"]
        a.plot([v.min(), v.max()], [y_, y_], color=ccol(c), lw=6, solid_capstyle="butt", alpha=.85)
        a.scatter(v, np.full(len(v), y_) + np.random.default_rng(1).uniform(-.18, .18, len(v)), s=4, color="k", alpha=.25, zorder=3)
        a.text(1820, y_, f"범위 {v.max() - v.min()}", va="center", fontsize=9)
    a.set_yticks(yy, [KO1[c] for c in order2], fontsize=9.5); a.set_xlim(1200, 1900)
    a.set_xlabel("Length_of_Conveyer (실제로는 강판 폭)"); a.set_title("③ Stains는 사실상 한 가지 폭에서만 나왔다")
    a = ax[1, 1]
    seen = Te.grp.isin(set(T.grp)); n_seen = seen.sum(); n_new = (~seen).sum()
    lab = ["Train에 있던\n공정 조건", "Train에 없던\n(처음 보는) 공정 조건"]
    err = [(74 - 7) / n_seen, 7 / 24]   # 07_final_model.ipynb 출력: 오분류 74건, 처음 보는 조건 24건 중 7건
    b = a.bar(lab, err, color=[EASY, BAD], width=.55)
    for r, e, n in zip(b, err, [n_seen, n_new]):
        a.text(r.get_x() + r.get_width() / 2, e + 0.006, f"오분류율 {e:.1%}\n(Test {n}건)", ha="center", fontsize=10)
    a.set_ylim(0, 0.4); a.set_ylabel("최종 모델 Test 오분류율"); a.set_title("④ 처음 보는 공정 조건에서는 더 자주 틀린다")
    fig.tight_layout(h_pad=2.5); save(fig, "process_condition")


def shape_similarity():
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    df, tr, te = load()
    SHAPE = ["LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index", "Square_Index", "Empty_Index",
             "Edges_X_Index", "Edges_Y_Index", "Luminosity_Index", "Minimum_of_Luminosity", "Maximum_of_Luminosity"]
    PROC = ["TypeOfSteel_A300", "Steel_Plate_Thickness", "Length_of_Conveyer"]
    lab = df.target.to_numpy()

    def ratio(cols):
        Z = StandardScaler().fit_transform(df[cols])
        _, ind = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z)
        return pd.Series((lab[ind[:, 1:]] == lab[:, None]).mean(1)).groupby(lab).mean()

    k = pd.DataFrame({"shape": ratio(SHAPE), "shape_proc": ratio(SHAPE + PROC)})
    # 최종 모델 Test F1 (07_final_model.ipynb 출력값)
    f1 = {"Other_Faults": 0.7714, "Bumps": 0.7421, "K_Scratch": 0.9536, "Z_Scratch": 0.9730,
          "Pastry": 0.5455, "Stains": 0.8889, "Dirtiness": 0.9524}
    order = ["K_Scratch", "Stains", "Other_Faults", "Bumps", "Dirtiness", "Pastry", "Z_Scratch"]
    x = np.arange(len(order)); w = .36
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.bar(x - w / 2, k.loc[order, "shape"], w, color="#9DB9D9", label="모양·밝기 특징 11개만 사용")
    ax.bar(x + w / 2, k.loc[order, "shape_proc"], w, color="#2E6FBA",
           label="+ 공정 조건 3개 (TypeOfSteel_A300, Steel_Plate_Thickness,\nLength_of_Conveyer)")
    ax.scatter(x, [f1[c] for c in order], marker="D", s=70, color=BAD, zorder=4, label="참고: 최종 모델 Test F1")
    for i, c in enumerate(order):
        ax.text(i - w / 2, k.loc[c, "shape"] + 0.015, f"{k.loc[c, 'shape']:.0%}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, k.loc[c, "shape_proc"] + 0.015, f"{k.loc[c, 'shape_proc']:.0%}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [KO[c] for c in order], fontsize=9.5); ax.set_ylim(0, 1.14)
    ax.set_ylabel("가장 닮은 결함 5개 중 같은 유형의 비율")
    ax.set_title("모양만 보면 헷갈리는 유형이 많다 — 특히 Z_Scratch는 공정 조건을 더해야 구분된다")
    ax.annotate("공정 조건을 더하면\n가장 크게 뛰는 유형", xy=(6 + w / 2, k.loc["Z_Scratch", "shape_proc"]), xytext=(4.7, 0.98),
                fontsize=9.5, arrowprops=dict(arrowstyle="->", color="#333"), ha="center")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.66, 1.02))
    fig.tight_layout(); save(fig, "shape_similarity")


def roll_mark_period():
    df, tr, te = load(); trs = set(tr)
    df["xc"] = (df.X_Minimum + df.X_Maximum) / 2
    col = df[(df.Length_of_Conveyer == 1358) & (df.Steel_Plate_Thickness == 40) & (df.TypeOfSteel_A300 == 0) & (abs(df.xc - 125) <= 20)]
    y = np.sort(col.Y_Minimum.values); p = 3548.5   # 05_team_verification.ipynb 4절에서 찾은 기본 주기
    gaps = np.diff(y); gaps = gaps[gaps > 0]; r = gaps / p

    def score(yy, pp, tol=15):
        d = np.diff(np.sort(yy)); d = d[d > 0]; rr = d / pp
        return np.mean(np.abs(rr - np.round(rr)) * pp <= tol)

    rng = np.random.default_rng(0)
    fixed = [score(rng.uniform(y.min(), y.max(), len(y)), p) for _ in range(200)]
    srch = []
    for _ in range(20):
        yy = rng.uniform(y.min(), y.max(), len(y)); srch.append(max(score(yy, pp) for pp in np.arange(1000, 20000, 2)))
    real = score(y, p)
    fig = plt.figure(figsize=(13.5, 8.4)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    a = fig.add_subplot(gs[0, :])
    seg = col.sort_values("Y_Minimum")
    intr = np.array([i in trs for i in seg.index])
    a.scatter(seg.Y_Minimum[intr], seg.xc[intr], s=22, color=EASY, label="Train에 들어간 결함")
    a.scatter(seg.Y_Minimum[~intr], seg.xc[~intr], s=40, color=BAD, marker="^", label="Test에 들어간 결함")
    a.set_xlabel("강판 길이 방향 위치 (Y_Minimum, 백만 단위)")
    a.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    a.set_ylabel("가로 위치 (X 중심)"); a.set_ylim(80, 170)
    a.set_title(f"① 같은 강판 묶음(폭 1358 · 두께 40 · A400)의 왼쪽 가장자리에 K_Scratch {len(seg)}건이 한 줄로 늘어서 있다")
    a.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2)
    b = fig.add_subplot(gs[1, 0])
    b.hist(r, bins=np.arange(0, 62, 0.25), color=EASY)
    for k_ in range(1, 31): b.axvline(k_, color=GREY, lw=.6, ls=":", zorder=0)
    b.set_xlim(0, 30); b.set_xlabel(f"이웃한 결함 사이 간격 ÷ 기본 주기({p:,.0f})"); b.set_ylabel("간격 수")
    b.set_title("② 간격이 기본 주기의 정수배 자리(점선)에만 몰려 있다")
    c = fig.add_subplot(gs[1, 1])
    vals = [real, np.mean(srch), np.mean(fixed)]
    lab = ["실제 데이터", "무작위 배치\n(주기도 다시 찾아줌)", "무작위 배치\n(같은 주기 적용)"]
    bb = c.bar(lab, vals, color=[BAD, GREY, "#C9CED4"], width=.55)
    for r_, v in zip(bb, vals):
        c.text(r_.get_x() + r_.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
    c.set_ylim(0, 1); c.set_ylabel("간격 중 주기의 정수배와 맞는 비율 (오차 ±15)")
    c.set_title("③ 우연으로는 나오기 어려운 규칙성")
    fig.tight_layout(h_pad=2.2); save(fig, "roll_mark_period")


def improvement_steps():
    # 값: Taegyeom_02~06 노트북과 07_final_model.ipynb 공식 출력값
    steps = ["① Baseline\nGBM 기본값", "② 불균형 처리\n클래스 가중치", "③ 하이퍼파라미터\n튜닝", "④ 파생변수 10개\n+ 재튜닝", "⑤ 앙상블\nET+RF+GBM"]
    v = [0.7958, 0.8095, 0.8225, 0.8401, 0.8441]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]; base = 0.78
    for i, (s, val) in enumerate(zip(steps, v)):
        lo = base if i == 0 else v[i - 1]
        a.bar(i, val - lo if i else val - base, bottom=lo if i else base, color=GREY if i == 0 else GOOD, width=.6)
        a.text(i, val + 0.002, f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
        if i: a.text(i, lo + (val - lo) / 2, f"+{val - lo:.4f}", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        if i < 4: a.plot([i + .3, i + .7], [val, val], color="#777", lw=.8, ls="--")
    a.set_xticks(range(5), steps, fontsize=9.3); a.set_ylim(base, 0.855); a.set_ylabel("Macro F1 (Train 5-fold CV)")
    a.set_title("① 무엇을 바꿨을 때 얼마나 올랐나 (최종 모델 경로)")
    a.text(4.35, 0.7815, "세로축이 0.78부터 시작함", ha="right", fontsize=8.5, color="#666")
    b = ax[1]
    names = ["LightGBM\n+ SMOTENC + 튜닝", "LightGBM\n+ 파생변수 + 재튜닝", "최종 모델\nET+RF+GBM"]
    cv5 = [0.8298, 0.8317, 0.8441]; rep = [0.8260, 0.8295, 0.8389]; test = [0.8252, 0.8226, 0.8324]
    x = np.arange(3); w = .26
    b.bar(x - w, cv5, w, color="#9DB9D9", label="5-fold CV (튜닝 때 본 점수)")
    b.bar(x, rep, w, color=EASY, label="반복 CV 5×3 (다른 분할로 다시 잼)")
    b.bar(x + w, test, w, color=HARD, label="Test (한 번도 안 본 데이터)")
    for i in range(3):
        for dx, val in zip([-w, 0, w], [cv5[i], rep[i], test[i]]): b.text(i + dx, val + 0.0015, f"{val:.3f}", ha="center", fontsize=8.3)
    b.set_xticks(x, names, fontsize=9.3); b.set_ylim(0.80, 0.855); b.set_ylabel("Macro F1")
    b.legend(frameon=False, fontsize=9, loc="upper left")
    b.text(2.4, 0.8008, "세로축이 0.80부터 시작함", ha="right", fontsize=8.5, color="#666")
    b.set_title("② 튜닝 점수는 부풀어 있지만, 최종 모델은 세 기준 모두 1위")
    fig.tight_layout(); save(fig, "improvement_steps")


def imbalance_comparison():
    r = cached("imbalance_repeated_cv.json"); cls = r["classes"]
    names = [("none", "처리 없음"), ("cw", "클래스 가중치"), ("bsmote", "BorderlineSMOTE"), ("smote", "SMOTE"), ("under", "언더샘플링")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    a = ax[0]; base = np.array(r["none"])
    for i, (k, n) in enumerate(names):
        v = np.array(r[k]); c = GREY if k == "none" else (BAD if k == "under" else EASY)
        a.bar(i, v.mean(), yerr=v.std(), color=c, width=.6, capsize=4)
        a.text(i, v.mean() + v.std() + 0.006, f"{v.mean():.3f}", ha="center", fontsize=10, fontweight="bold")
        if k != "none": a.text(i, 0.605, f"{(v > base).sum()}/15 우세", ha="center", fontsize=9, color="white")
    a.set_xticks(range(5), [n for _, n in names], fontsize=9.5); a.set_ylim(0.6, 0.86); a.set_ylabel("Macro F1 (GBM 기본값, 반복 CV 15번)")
    a.set_title("① 세 방법 모두 소폭 개선, 셋 사이 차이는 흔들림 안")
    a.text(4.45, 0.852, "세로축이 0.60부터 시작함", ha="right", fontsize=8.5, color="#666")
    b = ax[1]; order = ["Pastry", "Bumps", "Other_Faults", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness"]
    x = np.arange(len(order)); w = .38
    for j, (k, n, c) in enumerate([("cw", "클래스 가중치", EASY), ("bsmote", "BorderlineSMOTE", HARD)]):
        d = [r[k + "|rec"][cls.index(o)] - r["none|rec"][cls.index(o)] for o in order]
        b.bar(x + (j - .5) * w, d, w, color=c, label=n)
        for xx, v in zip(x + (j - .5) * w, d):
            b.text(xx, v + (0.004 if v >= 0 else -0.004), (f"{v:+.2f}" if abs(v) >= 0.005 else "0"),
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    b.axhline(0, color="#555", lw=1); b.set_xticks(x, [KO[o] for o in order], fontsize=8.6)
    b.set_ylabel("Recall 변화 (처리 없음 대비, 시드 42)"); b.legend(frameon=False, fontsize=9.5, loc="upper right")
    b.set_title("② Pastry·Bumps를 더 잡는 만큼 Other_Faults를 더 놓친다 (맞바꾸기)")
    fig.tight_layout(); save(fig, "imbalance_comparison")


def model_comparison():
    L = cached("cv_strat_vs_group.json")
    c = [("Baseline\nXGBoost", L["xgb"]["strat"]["mean"], L["xgb"]["group"]["mean"], GREY),
         ("GB + XGB\n+ BorderlineSMOTE", L["gbxgb"]["strat"]["mean"], L["gbxgb"]["group"]["mean"], HARD),
         ("LGBM + XGB + ET", None, None, "#9E9AC8"),
         ("최종 모델\nET + RF + GBM", L["final"]["strat"]["mean"], L["final"]["group"]["mean"], BAD),
         ("개선안: 최종 모델\n− plate_defect_count", L["nopdc"]["strat"]["mean"], L["nopdc"]["group"]["mean"], GOOD)]
    fig, ax = plt.subplots(figsize=(12.5, 5.6)); w = .36
    for i, (n, s, gg, col) in enumerate(c):
        if s is None:
            ax.text(i, 0.66, "반복 CV·그룹 CV\n미측정\n(5-fold CV 0.833)", ha="center", fontsize=9, color="#666"); continue
        ax.bar(i - w / 2, s, w, color=col, alpha=.45); ax.bar(i + w / 2, gg, w, color=col)
        ax.text(i - w / 2, s + 0.004, f"{s:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, gg + 0.004, f"{gg:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(c)), [x[0] for x in c], fontsize=9.5); ax.set_ylim(0.6, 0.88); ax.set_ylabel("Macro F1 (반복 교차검증 15번 평균)")
    ax.legend(handles=[mp.Patch(color=GREY, alpha=.45, label="연한 막대: 일반 CV (이미 본 생산 조건)"),
                       mp.Patch(color=GREY, label="진한 막대: 그룹 CV (처음 보는 생산 조건)")], frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("어떤 상황을 기준으로 하느냐에 따라 순위가 달라진다")
    ax.text(-0.45, 0.603, "세로축이 0.60부터 시작함 · 모든 값은 태겸 최종 코드 기준으로 다시 측정", ha="left", fontsize=8.5, color="#666")
    fig.tight_layout(); save(fig, "model_comparison")


def generalization():
    L = cached("cv_strat_vs_group.json")
    rows = [("공정 조건 3개만", "proconly"), ("공정 조건 제외\n(생김새·위치만)", "noproc"), ("Baseline\nXGBoost", "xgb"),
            ("최종 모델\n− 파생변수 전체", "nofe"), ("최종 모델", "final"), ("최종 모델\n− plate_defect_count", "nopdc")]
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]; w = .36
    for i, (n, m) in enumerate(rows):
        s, g = L[m]["strat"], L[m]["group"]
        c = BAD if m == "final" else (GOOD if m == "nopdc" else EASY)
        a.bar(i - w / 2, s["mean"], w, yerr=s["std"], color=c, alpha=.45, capsize=3, error_kw=dict(lw=.8))
        a.bar(i + w / 2, g["mean"], w, yerr=g["std"], color=c, capsize=3, error_kw=dict(lw=.8))
        a.text(i - w / 2, s["mean"] + s["std"] + 0.008, f"{s['mean']:.3f}", ha="center", fontsize=8.5)
        a.text(i + w / 2, g["mean"] + g["std"] + 0.008, f"{g['mean']:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    a.set_xticks(range(len(rows)), [n for n, _ in rows], fontsize=9); a.set_ylim(0.45, 0.92)
    a.set_ylabel("Macro F1 (반복 교차검증 15번 평균 ± 표준편차)")
    a.legend(handles=[mp.Patch(color=GREY, alpha=.45, label="연한 막대: 일반 CV (이미 본 생산 조건)"),
                      mp.Patch(color=GREY, label="진한 막대: 그룹 CV (처음 보는 생산 조건)")], frameon=False, fontsize=9, loc="upper left")
    a.set_title("① 모든 모델이 처음 보는 조건에서 크게 떨어진다")
    a.text(-0.6, 0.455, "세로축이 0.45부터 시작함", ha="left", fontsize=8.5, color="#666")
    b = ax[1]
    pairs = [("최종 − Baseline", "final", "xgb"), ("최종 − (파생변수 제외)", "final", "nofe"),
             ("(plate_defect_count 제외) − 최종", "nopdc", "final")]
    for j, (n, p, q) in enumerate(pairs):
        for k, (cn, col) in enumerate([("strat", EASY), ("group", BAD)]):
            d = np.array(L[p][cn]["folds"]) - np.array(L[q][cn]["folds"])
            yy = j * 2.6 + k * 1.0
            b.scatter(d, np.full(len(d), yy) + np.random.default_rng(j * 3 + k).uniform(-.18, .18, len(d)), s=22, color=col, alpha=.7)
            b.plot([d.mean()] * 2, [yy - .35, yy + .35], color="k", lw=2.2)
            b.text(0.112, yy, f"{'일반' if cn == 'strat' else '그룹'} CV  평균 {d.mean():+.3f}  ({(d > 0).sum()}/15 우세)", va="center", fontsize=8.8)
        b.text(-0.112, j * 2.6 + 0.5, n, va="center", ha="right", fontsize=9.5, fontweight="bold")
    b.axvline(0, color="#555", lw=1); b.set_yticks([]); b.set_xlim(-0.1, 0.1)
    b.spines["left"].set_visible(False); b.set_xlabel("같은 fold끼리 점수 차이 (점 하나 = fold 하나)")
    b.set_title("② 파생변수의 효과는 두 기준에서 정반대")
    fig.tight_layout(); save(fig, "generalization")


def error_structure():
    from sklearn.metrics import confusion_matrix
    df, tr, te = load()
    o = cached("final_model_oof_5fold.csv")
    y = df.loc[o["index"], "target"].values; oof = o["pred"].values
    order = ["K_Scratch", "Z_Scratch", "Stains", "Dirtiness", "Bumps", "Pastry", "Other_Faults"]
    cm = confusion_matrix(y, oof, labels=order); cmn = cm / cm.sum(1, keepdims=True)
    six = 0.9067   # 05_team_verification.ipynb 8절: Other_Faults 제외 6종 5-fold
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    a = ax[0]; a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            if cm[i, j]: a.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8.3,
                                color="white" if cmn[i, j] > .55 else "#222")
    a.set_xticks(range(7), order, fontsize=8.3); a.set_yticks(range(7), order, fontsize=8.3)
    a.set_xlabel("모델이 예측한 유형"); a.set_ylabel("실제 유형")
    a.add_patch(mp.Rectangle((3.5, 3.5), 3, 3, fill=False, ec=HARD, lw=3))
    a.set_title("① 오류 대부분이 Bumps·Pastry·Other_Faults(주황 테두리) 안에서 돈다", pad=12)
    inside = cm[4:, 4:].sum() - np.trace(cm[4:, 4:]); tot = cm.sum() - np.trace(cm)
    a.text(3, 8.05, f"전체 오류 {tot}건 중 세 유형끼리의 오류 {inside}건 ({inside / tot:.0%}) · 최종 모델 Train 5-fold CV",
           ha="center", fontsize=9, color="#444")
    b = ax[1]; v = [0.8432, six]
    bb = b.bar(["일곱 유형 전체", "Other_Faults를 뺀 여섯 유형"], v, color=[GREY, GOOD], width=.5)
    for r, x in zip(bb, v): b.text(r.get_x() + r.get_width() / 2, x + 0.004, f"{x:.3f}", ha="center", fontsize=12, fontweight="bold")
    b.set_ylim(0.7, 0.95); b.set_ylabel("Macro F1 (Train 5-fold CV, 같은 모델·같은 설정)")
    b.text(-0.3, 0.703, "세로축이 0.70부터 시작함", ha="left", fontsize=8.5, color="#666")
    b.set_title("② Other_Faults를 빼면 점수가 크게 뛴다", pad=12)
    fig.tight_layout(); save(fig, "error_structure")


def confusion_matrix_fig():
    # 07_final_model.ipynb 공식 출력값 (Test 389건)
    ORDER = ["Other_Faults", "Bumps", "K_Scratch", "Z_Scratch", "Pastry", "Stains", "Dirtiness"]
    cm = np.array([[108, 14, 1, 0, 11, 1, 0], [18, 59, 0, 0, 4, 0, 0], [5, 0, 72, 0, 1, 0, 0], [2, 0, 0, 36, 0, 0, 0],
                   [10, 4, 0, 0, 18, 0, 0], [1, 1, 0, 0, 0, 12, 0], [1, 0, 0, 0, 0, 0, 10]])
    pr = [.745, .756, .986, 1.000, .529, .923, 1.000]; rc = [.800, .728, .923, .947, .562, .857, .909]
    f1v = [.771, .742, .954, .973, .545, .889, .952]; sup = [135, 81, 78, 38, 32, 14, 11]
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]; a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            if cm[i, j]: a.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8.5,
                                color="white" if cmn[i, j] > .55 else "#222")
    a.set_xticks(range(7), ORDER, fontsize=8.2, rotation=30, ha="right"); a.set_yticks(range(7), ORDER, fontsize=8.2)
    a.set_xlabel("모델이 예측한 유형"); a.set_ylabel("실제 유형")
    a.add_patch(mp.Rectangle((-0.5, -0.5), 2, 2, fill=False, ec=HARD, lw=2.5))
    a.add_patch(mp.Rectangle((3.5, 3.5), 1, 1, fill=False, ec=HARD, lw=2.5))
    a.set_title("① Test 혼동행렬 — 오류가 주황 테두리 안에 몰려 있다", pad=10)
    b = ax[1]; y = np.arange(7)[::-1]; w = .38
    b.barh(y + w / 2, pr, w, color="#9DB9D9", label="Precision (그 유형이라 했을 때 맞을 확률)")
    b.barh(y - w / 2, rc, w, color=EASY, label="Recall (실제 그 유형을 놓치지 않는 정도)")
    for yy, p_, r_ in zip(y, pr, rc):
        b.text(p_ + 0.012, yy + w / 2, f"{p_:.2f}", va="center", fontsize=8.5); b.text(r_ + 0.012, yy - w / 2, f"{r_:.2f}", va="center", fontsize=8.5)
    b.set_yticks(y, [f"{c}\nF1 {f:.2f} · {s}건" for c, f, s in zip(ORDER, f1v, sup)], fontsize=8.2)
    b.set_xlim(0, 1.18); b.set_xlabel("Test 성능"); b.legend(frameon=False, fontsize=8.8, loc="lower right")
    b.set_title("② Pastry만 두 지표가 함께 낮다", pad=10)
    fig.tight_layout(); save(fig, "confusion_matrix")


def learning_curve():
    # 07_final_model.ipynb 4-4 셀 공식 출력값
    n = [248, 496, 745, 993, 1242]
    vf1 = [0.7430, 0.7753, 0.8078, 0.8263, 0.8441]; vls = [0.6907, 0.5890, 0.5482, 0.5120, 0.4827]
    tls = [0.0587, 0.0545, 0.0546, 0.0547, 0.0571]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    a = ax[0]; a.plot(n, [1.0] * 5, "o-", color=GREY, label="학습 데이터"); a.plot(n, vf1, "o-", color=EASY, lw=2.4, label="검증 데이터")
    a.fill_between(n, vf1, [1.0] * 5, color=GREY, alpha=.12)
    for x_, v in zip(n, vf1): a.text(x_, v - 0.022, f"{v:.3f}", ha="center", fontsize=8.5)
    a.set_ylim(0.6, 1.05); a.set_xlabel("학습에 쓴 결함 수"); a.set_ylabel("Macro F1")
    a.set_title("① 데이터를 늘릴수록 검증 점수가 계속 오른다"); a.legend(frameon=False, fontsize=9.5, loc="center right")
    a.text(1242, 0.93, "회색 간격 = 학습과 검증의 차이\n(데이터가 늘수록 좁아짐)", ha="right", fontsize=9, color="#555")
    b = ax[1]; b.plot(n, tls, "o-", color=GREY, label="학습 데이터"); b.plot(n, vls, "o-", color=BAD, lw=2.4, label="검증 데이터")
    for x_, v in zip(n, vls): b.text(x_, v + 0.022, f"{v:.3f}", ha="center", fontsize=8.5)
    b.set_ylim(0, 0.78); b.set_xlabel("학습에 쓴 결함 수"); b.set_ylabel("Log Loss (확률이 틀린 정도, 낮을수록 좋음)")
    b.set_title("② 확률의 정확도도 계속 좋아진다"); b.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.tight_layout(); save(fig, "learning_curve")


def loss_curve():
    """교차검증 한 fold에서 직접 계산 (약 30초)."""
    from sklearn.metrics import f1_score, log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
    df, tr, te = load()
    X_train = df.loc[tr].drop(columns="target").reset_index(drop=True); y_train = df.loc[tr, "target"].reset_index(drop=True)
    PLATE_KEYS = ["Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300"]
    DISCRETE = ["TypeOfSteel_A300", "Outside_Global_Index"]
    plate_key = lambda X: X[PLATE_KEYS].astype(str).agg("_".join, axis=1)

    def add_features(X, plate_counts):   # 07_final_model.ipynb와 같은 파생변수
        out = X.copy(); width = (X["X_Maximum"] - X["X_Minimum"]).clip(lower=0)
        mx, mn = X["Maximum_of_Luminosity"], X["Minimum_of_Luminosity"]
        out["compactness"] = (X["X_Perimeter"] + X["Y_Perimeter"]) / np.sqrt(X["Pixels_Areas"])
        out["perimeter_ratio_log"] = np.log((X["X_Perimeter"] + 1) / (X["Y_Perimeter"] + 1))
        out["x_straightness"] = X["X_Perimeter"] / (width + 1)
        out["lum_range"] = mx - mn
        out["lum_contrast"] = (mx - mn) / (mx + mn).replace(0, 1)
        out["lum_mean_position"] = (X["Sum_of_Luminosity"] / X["Pixels_Areas"] - mn) / (mx - mn + 1)
        xc = (X["X_Minimum"] + X["X_Maximum"]) / 2 / X["Length_of_Conveyer"]
        out["x_center_rel"] = xc; out["x_center_dist"] = (xc - 0.5).abs()
        out["plate_defect_count"] = plate_key(X).map(plate_counts).fillna(0).astype(float)
        out["thickness_per_conveyer"] = X["Steel_Plate_Thickness"] / X["Length_of_Conveyer"]
        return out[[c for c in out.columns if c not in DISCRETE] + DISCRETE]

    tr_i, va_i = next(iter(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_train, y_train)))
    Xtr, Xva = X_train.iloc[tr_i], X_train.iloc[va_i]; ytr, yva = y_train.iloc[tr_i], y_train.iloc[va_i]
    pc = plate_key(Xtr).value_counts().to_dict(); Ftr, Fva = add_features(Xtr, pc), add_features(Xva, pc)
    cls = np.unique(y_train); cnt = ytr.value_counts()
    gbm = GradientBoostingClassifier(n_estimators=240, learning_rate=0.08793919399591556, max_depth=4, subsample=0.9079017872741743,
                                     max_features=0.5, min_samples_leaf=5, random_state=RS)
    gbm.fit(Ftr, ytr, sample_weight=ytr.map(cnt.max() / cnt).to_numpy())
    gstage = list(gbm.staged_predict_proba(Fva))
    gloss = [log_loss(yva, p, labels=list(cls)) for p in gstage]; gf1 = [f1_score(yva, cls[p.argmax(1)], average="macro") for p in gstage]
    ck = np.linspace(0.05, 1.0, 12)
    et = ExtraTreesClassifier(max_depth=15, max_features=0.7, class_weight="balanced", random_state=RS, warm_start=True, n_jobs=1)
    rf = RandomForestClassifier(max_features=0.3, random_state=RS, warm_start=True, n_jobs=1)
    eloss, rloss, ens = [], [], []
    for c in ck:
        et.set_params(n_estimators=max(5, int(438 * c))); et.fit(Ftr, ytr)
        rf.set_params(n_estimators=max(5, int(483 * c))); rf.fit(Ftr, ytr)
        pe, prf = et.predict_proba(Fva), rf.predict_proba(Fva); pg = gstage[max(0, int(240 * c) - 1)]
        eloss.append(log_loss(yva, pe, labels=list(cls))); rloss.append(log_loss(yva, prf, labels=list(cls)))
        ens.append(log_loss(yva, (pe + prf + pg) / 3, labels=list(cls)))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    a = ax[0]; x = np.arange(1, 241)
    a.plot(x, gloss, color=BAD, lw=2, label="GBM 혼자")
    a.axvline(int(np.argmin(gloss)) + 1, color=BAD, ls=":", lw=1)
    a.text(int(np.argmin(gloss)) + 6, max(gloss) * 0.97, f"여기서부터 다시 나빠짐\n(트리 {int(np.argmin(gloss)) + 1}개)", fontsize=9, color=BAD)
    a2 = a.twinx(); a2.plot(x, gf1, color=EASY, lw=1.6, alpha=.85); a2.set_ylabel("Macro F1 (파란선)", color=EASY); a2.tick_params(colors=EASY)
    a.set_xlabel("쌓은 트리 수"); a.set_ylabel("검증 Log Loss (빨간선)", color=BAD); a.tick_params(axis="y", colors=BAD)
    a.set_title("① GBM 혼자: 손실은 올라가도 F1은 안 떨어진다", pad=10)
    b = ax[1]
    b.plot(ck * 100, eloss, "o-", color="#9DB9D9", lw=1.6, ms=4, label="ExtraTrees")
    b.plot(ck * 100, rloss, "o-", color="#C9A227", lw=1.6, ms=4, label="Random Forest")
    b.plot(ck * 100, [gloss[max(0, int(240 * c) - 1)] for c in ck], "o-", color=BAD, lw=1.6, ms=4, label="GBM")
    b.plot(ck * 100, ens, "o-", color=GOOD, lw=2.8, ms=5, label="세 모델 평균 (최종 모델)")
    b.set_xlabel("학습 진행도 (%)"); b.set_ylabel("검증 Log Loss"); b.legend(frameon=False, fontsize=9)
    b.set_title("② 묶으면 GBM의 과신이 거의 상쇄된다", pad=10)
    b.text(100, min(ens) - 0.045, "끝까지 거의 평평\n(GBM 혼자일 때의 상승이 사라짐)", ha="right", fontsize=9, color=GOOD)
    b.text(5, 0.90, "교차검증 한 fold 기준", fontsize=8.5, color="#666")
    fig.tight_layout(); save(fig, "loss_curve")


def feature_importance():
    d = cached("permutation_importance_test.csv").sort_values("mean", ascending=False).head(14)[::-1]
    PROC = ["Steel_Plate_Thickness", "Length_of_Conveyer", "TypeOfSteel_A300"]
    POS = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum"]
    col = [BAD if f in PROC else (GREY if f in POS else EASY) for f in d.f]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.barh(range(len(d)), d["mean"], xerr=d["std"], color=col, capsize=2.5, error_kw=dict(lw=.7))
    for i, v in enumerate(d["mean"]):
        ax.text(v + d["std"].iloc[i] + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(d)), d.f, fontsize=9); ax.set_xlim(0, 0.19)
    ax.set_xlabel("그 변수를 뒤섞었을 때 Test Macro F1이 떨어지는 정도 (3회 평균 ± 표준편차)")
    ax.legend(handles=[mp.Patch(color=BAD, label="공정 조건 (어느 강판에서 나왔나)"), mp.Patch(color=EASY, label="결함 자체의 측정값"),
                       mp.Patch(color=GREY, label="결함의 위치")], frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("모델이 가장 많이 쓰는 정보는 결함의 생김새가 아니라 \"어느 강판인가\"")
    ax.text(0.19, -1.75, "원본 26개 열 기준 · 파생변수는 이 열들로 계산되므로 파생변수의 기여도 함께 반영됨", ha="right", fontsize=8.5, color="#666")
    fig.tight_layout(); save(fig, "feature_importance")


FIGURES = {
    "overview": overview, "target_distribution": target_distribution, "eda_result": eda_result,
    "process_condition": process_condition, "shape_similarity": shape_similarity, "roll_mark_period": roll_mark_period,
    "improvement_steps": improvement_steps, "imbalance_comparison": imbalance_comparison,
    "model_comparison": model_comparison, "generalization": generalization, "error_structure": error_structure,
    "confusion_matrix": confusion_matrix_fig, "learning_curve": learning_curve, "loss_curve": loss_curve,
    "feature_importance": feature_importance,
}


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="README 그림 재생성 (기본: figures/regenerated/ 에 저장)")
    ap.add_argument("--only", default="", help="만들 그림 이름 (쉼표로 구분)")
    ap.add_argument("--list", action="store_true", help="그림 이름 목록만 출력")
    ap.add_argument("--overwrite", action="store_true", help="figures/ 의 README 그림을 덮어씀 (먼저 백업)")
    args = ap.parse_args()
    if args.list:
        print("\n".join(FIGURES)); return
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        sys.exit(f"[오류] 없는 그림 이름: {unknown}  (--list 로 확인)")
    if args.overwrite:
        OUT_DIR = FIG_DIR
        backup = FIG_DIR / f"_backup_{datetime.now():%Y%m%d_%H%M%S}"
        backup.mkdir(parents=True)
        for n in names:
            if (FIG_DIR / f"{n}.png").exists():
                shutil.copy2(FIG_DIR / f"{n}.png", backup / f"{n}.png")
        print(f"[덮어쓰기] 기존 그림을 {backup.relative_to(ROOT)} 에 백업했습니다.")
    else:
        OUT_DIR = FIG_DIR / "regenerated"
        print("[안내] README 그림은 그대로 두고 figures/regenerated/ 에 저장합니다.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        print(f"- {n}")
        FIGURES[n]()


if __name__ == "__main__":
    main()
