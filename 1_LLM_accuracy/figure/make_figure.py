#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ======== INPUTS ========
CSV_PATHS = [
    Path("/root/mnt/RangeGuard_AE/1_LLM_accuracy/figure/source_csv/Llama-3.2-1B.csv"),
    Path("/root/mnt/RangeGuard_AE/1_LLM_accuracy/figure/source_csv/Llama-3.1-8B.csv"),
]
OUT_DIR   = Path("/root/mnt/RangeGuard_AE/1_LLM_accuracy/figure/output")
# ========================

# 높이만 줄임 (7 -> 5)
W, H = 28.0, 5.0
plt.rcParams.update({
    "figure.figsize": (W, H),
    "figure.dpi": 125,
    "savefig.dpi": 125,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    # 범례 크기 줄임 (30 -> 24)
    "legend.fontsize": 24,
    "legend.title_fontsize": 20,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

pretty = {
    "RG_4b_DSC": "RG 4b DSC",
    "RG_8b_SSC": "RG 8b SSC",
    "VAPI": "VAPI",
    "Weight_Nulling": "Weight Nulling",
    "no_protection": "No protection",
}

desired_scheme_order = [
    "no_protection",
    "Weight_Nulling",
    "VAPI",
    "RG_8b_SSC",
    "RG_4b_DSC",
]

scheme_to_slot = {
    "no_protection":  2,
    "Weight_Nulling": 3,
    "VAPI":           4,
    "RG_8b_SSC":      5,
    "RG_4b_DSC":      6,
}

fixed_colors = {
    "no_protection":  "#4d4d4d",
    "Weight_Nulling": "#7f7f7f",
    "VAPI":           "#1f77b4",
    "RG_4b_DSC":      "#ff7f0e",
    "RG_8b_SSC":      "#d62728",
}

NOERR_SLOTS, GROUP_SLOTS, NOERR_CENTER = 5, 7, 3

def group_start_slot(j: int) -> int:
    return NOERR_SLOTS + 1 + j * GROUP_SLOTS

def normalize_ber(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.strip().str.lower()
         .str.replace(r"\s+", "", regex=True)
         .replace({
             "baseline": "noerror", "no-error": "noerror", "noerror": "noerror", "no_error": "noerror",
             "0": "noerror", "0.0": "noerror", "none": "noerror"
         }))
    s = s.where(s != "noerror", "no error")

    def canon(v: str) -> str:
        if v == "no error":
            return v
        try:
            f = float(v)
        except Exception:
            return v
        if f <= 0:
            return v
        exp = int(round(-np.log10(f)))
        if np.isfinite(exp) and np.isclose(f, 10.0 ** (-exp), rtol=0, atol=1e-12):
            return f"1e-{exp}"
        return v

    return s.map(canon)

def make_plot(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not {"scheme", "acc_none"}.issubset(df.columns):
        raise ValueError(f"{csv_path}: missing required columns")

    d = df[~df["acc_none"].isna()].copy()
    if "BER" in d.columns:
        d["BER"] = normalize_ber(d["BER"])
    elif "ber_value" in d.columns and d["ber_value"].notna().any():
        d["BER"] = normalize_ber(d["ber_value"])
    elif "ber_exp" in d.columns:
        d["BER"] = normalize_ber(d["ber_exp"].apply(
            lambda e: f"1e-{int(e)}" if str(e).isdigit() else "no error"))
    else:
        raise ValueError(f"{csv_path}: missing BER info")

    if d["acc_none"].max() > 1.1:
        d["acc_none"] /= 100.0

    ber_order = ["no error", "1e-10", "1e-9", "1e-8",
                 "1e-7", "1e-6", "1e-5", "1e-4"]
    present_bers = [b for b in ber_order if b in set(d["BER"])] or list(d["BER"].unique())

    d["scheme"] = d["scheme"].astype(str).str.strip()
    present_schemes = [s for s in desired_scheme_order if s in set(d["scheme"])]
    if not present_schemes:
        raise ValueError(f"{csv_path}: no matching schemes")

    fig, ax = plt.subplots()
    legend_handles, legend_names, seen = [], [], set()

    has_noerror = "no error" in set(d["BER"])
    xticks = [NOERR_CENTER] if has_noerror else []
    xticklabels = ["no error"] if has_noerror else []
    others = [b for b in present_bers if b != "no error"]
    for j, ber in enumerate(others):
        s0 = group_start_slot(j)
        center = s0 + 3
        xticks.append(center)
        xticklabels.append(ber)

    arr_noerr = d[d["BER"] == "no error"]["acc_none"].values
    if has_noerror and arr_noerr.size > 0:
        y_med = float(np.median(arr_noerr))
        ax.scatter([NOERR_CENTER], [y_med],
                   marker="D", s=100,
                   facecolor="#4d4d4d", edgecolor="black",
                   linewidth=1.2, alpha=0.95, zorder=5)

    box_width = 0.85
    for j, ber in enumerate(others):
        s0 = group_start_slot(j)
        arrays, positions, labels_local = [], [], []
        for scheme in present_schemes:
            vals = d[(d["BER"] == ber) & (d["scheme"] == scheme)]["acc_none"].values
            if vals.size == 0:
                continue
            x_pos = s0 + (scheme_to_slot.get(scheme, 0) - 1)
            arrays.append(vals)
            positions.append(x_pos)
            labels_local.append(scheme)

        if not arrays:
            continue
        bp = ax.boxplot(arrays, positions=positions, widths=box_width,
                        manage_ticks=False, patch_artist=True,
                        whis=(0, 100), showfliers=False)
        for k, scheme in enumerate(labels_local):
            c = fixed_colors.get(scheme, "#999999")
            bp["boxes"][k].set(facecolor=c, edgecolor=c, alpha=0.8, linewidth=2.0)
            for part in ["medians", "whiskers", "caps"]:
                if part == "medians":
                    lines = [bp[part][k]]
                else:
                    lines = bp[part][2*k:2*k+2]
                for line in lines:
                    line.set(color=c, linewidth=2.0)
            if scheme not in seen:
                legend_handles.append(Patch(facecolor=c, edgecolor=c,
                                            label=pretty.get(scheme, scheme)))
                legend_names.append(pretty.get(scheme, scheme))
                seen.add(scheme)

    if has_noerror:
        ax.axvline(x=NOERR_SLOTS + 0.5, color="gray",
                   linestyle="--", linewidth=0.8, alpha=0.4)
    for j, _ in enumerate(others):
        s0 = group_start_slot(j)
        ax.axvline(x=s0 + GROUP_SLOTS - 0.5, color="gray",
                   linestyle="--", linewidth=0.8, alpha=0.4)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)

    if others:
        last_s0 = group_start_slot(len(others)-1)
        x_right = last_s0 + GROUP_SLOTS - 0.5
        x_left = 0.5 if has_noerror else group_start_slot(0) - 0.5
    else:
        x_right = NOERR_SLOTS + 0.5 if has_noerror else 7.5
        x_left = 0.5 if has_noerror else 6.5
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("BER")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)

    # y축 틱을 0.2 간격으로 설정 → 이에 맞춰 가로 그리드가 생김
    yticks = np.arange(0.0, 1.01, 0.2)
    ax.set_yticks(yticks)

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # 범례를 조금 더 위로 올려서 데이터와 겹치지 않게
    leg = ax.legend(legend_handles, legend_names,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.12),  # 1.00 -> 1.12 정도로 살짝 위
                    ncols=len(legend_names),
                    frameon=True, handlelength=1.5,
                    columnspacing=1.2, handletextpad=0.6,
                    fancybox=False,
                    borderaxespad=0.5)
    frame = leg.get_frame()
    frame.set_linewidth(1.0)
    frame.set_edgecolor("#666666")
    frame.set_facecolor("white")
    frame.set_alpha(1.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = csv_path.stem
    plt.savefig(OUT_DIR / f"{model}_ecc.png")
    plt.savefig(OUT_DIR / f"{model}_ecc.pdf")
    plt.close(fig)
    print(f"Saved figure for {model}")

def main():
    for path in CSV_PATHS:
        make_plot(path)

if __name__ == "__main__":
    main()
