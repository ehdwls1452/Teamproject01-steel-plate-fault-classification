"""혼동행렬 시각화. 노트북과 Streamlit 대시보드에서 공용으로 사용한다."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=True,
                          title="혼동행렬", save_name=None, ax=None):
    """
    y_true, y_pred : 실제값, 예측값
    normalize      : True면 실제 클래스별 비율(재현율), False면 건수
    save_name      : 파일명을 주면 figures/ 에 저장
    ax             : 기존 축에 그릴 때 전달 (대시보드용)
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        with np.errstate(all="ignore"):
            shown = cm / cm.sum(axis=1, keepdims=True)
        shown = np.nan_to_num(shown)
        fmt = lambda i, j: f"{shown[i, j]:.2f}\n({cm[i, j]})"
        cbar_label = "실제 대비 비율"
    else:
        shown = cm
        fmt = lambda i, j: f"{cm[i, j]}"
        cbar_label = "건수"

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    else:
        fig = ax.figure

    im = ax.imshow(shown, cmap="Blues", vmin=0,
                   vmax=1 if normalize else shown.max())

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("모델의 예측")
    ax.set_ylabel("실제 결함")
    ax.set_title(title, fontsize=13, pad=12)

    # 배경이 어두우면 글자를 흰색으로
    limit = (1 if normalize else shown.max()) * 0.55
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, fmt(i, j), ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > limit else "#222")

    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)

    if created:
        plt.tight_layout()
    if save_name:
        out = BASE_DIR / "figures" / save_name
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"저장: {out}")
    return fig, ax, cm