#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = REPO_ROOT / "1_LLM_accuracy" / "output"
DEFAULT_OUT_DIR = REPO_ROOT / "1_LLM_accuracy" / "figure" / "source_csv"

SCHEME_MAP = {
    "NONE": "no_protection",
    "NULLING_BF16": "Weight_Nulling",
    "VAPI": "VAPI",
    "RANGEGUARD_BF16_SSC_FAST": "RG_8b_SSC",
    "RANGEGUARD_BF16_DSC_FAST": "RG_4b_DSC",
}

def find_acc_none(results_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(results_obj, dict):
        return (None, None)
    for task, metrics in results_obj.items():
        if isinstance(metrics, dict) and 'acc,none' in metrics:
            try:
                return (task, float(metrics.get('acc,none')))
            except (TypeError, ValueError):
                return (task, None)
    return (None, None)

def parse_path_components(path: str, base_dir: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    반환: (model, task, scheme, ber_exp, ber_label)
    - exp_<N>  -> ber_exp=N, ber_label='1e-<N>'
    - baseline_* 또는 그 외 -> ber_exp=None, ber_label='baseline'
    """
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    # expected: <model>/<task>/<scheme>/<exp_x>/.../*.json
    if len(parts) < 4:
        return (None, None, None, None, None)
    model, task, scheme_raw, exp_dir = parts[0], parts[1], parts[2], parts[3]
    scheme = SCHEME_MAP.get(scheme_raw, scheme_raw)

    if exp_dir.startswith('exp_'):
        try:
            ber_exp = int(exp_dir.split('_', 1)[1])
        except ValueError:
            ber_exp = None
        ber_label = f'1e-{ber_exp}' if isinstance(ber_exp, int) and ber_exp >= 0 else 'baseline'
        return (model, task, scheme, ber_exp, ber_label)

    # baseline 폴더(예: baseline_none) 포함: 전부 baseline 처리
    return (model, task, scheme, None, 'baseline')

def collect_json_files(base_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith('.json'):
                out.append(os.path.join(root, fn))
    return out

def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['model','scheme','BER','ber_exp','ber_value','task','acc_none','json_path']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    p = argparse.ArgumentParser(description="Aggregate acc,none from result JSONs into per-model CSVs.")
    p.add_argument('--base-dir', default=str(DEFAULT_BASE_DIR), help='Root directory to scan')
    p.add_argument('--out-dir',  default=str(DEFAULT_OUT_DIR), help='Directory to write CSV files')
    p.add_argument('--models', nargs='*', default=None, help='Whitelist of model names (e.g., Llama-3.2-1B ResNet-50)')
    p.add_argument('--require-acc-none', dest='require_acc_none', action='store_true', help='Skip JSONs without acc,none')
    args = p.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_dir = os.path.abspath(args.out_dir)

    paths = collect_json_files(base_dir)
    if not paths:
        print(f'[json2csv] No JSON files found under {base_dir}')
        return

    print(f'[json2csv] Found {len(paths)} JSON files, scanning...')

    per_model_rows: Dict[str, List[Dict[str, Any]]] = {}
    skipped_parse_error = 0
    skipped_unsupported_top_level = 0

    for jp in paths:
        model, task_from_path, scheme, ber_exp, ber_label = parse_path_components(jp, base_dir)
        if model is None or scheme is None:
            continue
        if args.models and model not in args.models:
            continue

        ber_value = None
        if isinstance(ber_exp, int) and ber_exp >= 0:
            ber_value = 10.0 ** (-ber_exp)

        try:
            with open(jp, 'r') as f:
                obj = json.load(f)
        except Exception as e:
            print(f'[warn] Failed to parse {jp}: {e}')
            skipped_parse_error += 1
            continue

        # Some result dumps are arrays or other top-level types.
        # Skip unsupported shapes instead of crashing.
        if not isinstance(obj, dict):
            skipped_unsupported_top_level += 1
            continue

        results_obj = obj.get('results')
        task, acc_none = find_acc_none(results_obj) if results_obj else (None, None)
        if args.require_acc_none and acc_none is None:
            continue

        row = {
            'model': model,
            'scheme': scheme,
            'BER': ber_label,                                           # ← 문자열 BER: 'baseline' 또는 '1e-<N>'
            'ber_exp': ber_exp,                                         # 예: 4
            'ber_value': f'{ber_value:.0e}' if ber_value is not None else None,  # 예: '1e-04'
            'task': task if task is not None else task_from_path,
            'acc_none': acc_none,
            'json_path': jp,
        }
        per_model_rows.setdefault(model, []).append(row)

    # write per-model CSVs only
    os.makedirs(out_dir, exist_ok=True)
    for model, rows in per_model_rows.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r['scheme'] or '',
                r['ber_exp'] if isinstance(r.get('ber_exp'), int) else -1,
                r['json_path'] or '',
            ),
        )
        out_path = os.path.join(out_dir, f'{model}.csv')
        write_csv(rows_sorted, out_path)
        print(f'[json2csv] wrote {out_path} ({len(rows_sorted)} rows)')

    print(
        f'[json2csv] done: parse_error_skipped={skipped_parse_error}, '
        f'unsupported_top_level_skipped={skipped_unsupported_top_level}'
    )

if __name__ == '__main__':
    main()
