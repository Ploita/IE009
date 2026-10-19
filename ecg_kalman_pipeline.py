#!/usr/bin/env python3
"""Pipeline para remoção de ruído em ECG via filtro de Kalman e avaliação ML."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - dependência opcional em runtime
    wfdb = None
    WFDB_IMPORT_ERROR = exc
else:
    WFDB_IMPORT_ERROR = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}


@dataclass
class RecordEvaluation:
    record: str
    snr_noisy_db: float
    snr_filtered_db: float


def kalman_filter_ecg(signal: np.ndarray, q: float = 1e-5, r: float = 2e-3) -> np.ndarray:
    """Filtra um ECG 1D usando um Kalman linear com modelo de velocidade constante."""
    if signal.ndim != 1:
        raise ValueError("O sinal deve ser unidimensional")
    if signal.size == 0:
        return signal.copy()

    a = np.array([[1.0, 1.0], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q_cov = q * np.eye(2)
    r_cov = np.array([[r]])

    x = np.array([signal[0], 0.0])
    p = np.eye(2)

    filtered = np.empty_like(signal, dtype=float)

    for idx, measurement in enumerate(signal):
        x_pred = a @ x
        p_pred = a @ p @ a.T + q_cov

        innovation = np.array([measurement]) - (h @ x_pred)
        s = h @ p_pred @ h.T + r_cov
        k = p_pred @ h.T @ np.linalg.inv(s)

        x = x_pred + (k @ innovation).ravel()
        p = (np.eye(2) - k @ h) @ p_pred

        filtered[idx] = x[0]

    return filtered


def compute_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Calcula SNR(reference, estimate) em dB, onde erro = reference - estimate."""
    if reference.shape != estimate.shape:
        raise ValueError("reference e estimate precisam ter o mesmo formato")

    signal_power = float(np.mean(np.square(reference)))
    noise_power = float(np.mean(np.square(reference - estimate)))

    if noise_power == 0.0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def add_noise_at_target_snr(clean_signal: np.ndarray, noise_signal: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Adiciona ruído ao sinal limpo mantendo SNR alvo."""
    if clean_signal.size == 0:
        return clean_signal.copy()

    n = min(clean_signal.size, noise_signal.size)
    clean = clean_signal[:n]
    noise = noise_signal[:n]

    clean_power = float(np.mean(np.square(clean)))
    noise_power = float(np.mean(np.square(noise)))
    if noise_power == 0.0:
        return clean.copy()

    target_linear = 10.0 ** (target_snr_db / 10.0)
    scaling = np.sqrt(clean_power / (target_linear * noise_power))

    return clean + scaling * noise


def beat_to_binary(symbol: str) -> int:
    return 0 if symbol in NORMAL_BEAT_SYMBOLS else 1


def _moment_ratio(centered: np.ndarray, power: int) -> float:
    return float(np.mean(np.power(centered, power)))


def build_features_from_segment(segment: np.ndarray, pre_rr: float, post_rr: float, peak_value: float) -> np.ndarray:
    centered = segment - np.mean(segment)
    variance = float(np.var(segment))
    std = float(np.sqrt(variance))

    skewness = _moment_ratio(centered, 3) / (std**3 + 1e-12)
    kurtosis = _moment_ratio(centered, 4) / (std**4 + 1e-12)

    return np.array(
        [
            float(np.mean(segment)),
            std,
            float(np.min(segment)),
            float(np.max(segment)),
            float(np.ptp(segment)),
            float(np.median(segment)),
            float(np.mean(np.abs(np.diff(segment)))),
            float(np.mean(np.square(segment))),
            skewness,
            kurtosis,
            pre_rr,
            post_rr,
            peak_value,
        ],
        dtype=float,
    )


def extract_paired_features(
    noisy_signal: np.ndarray,
    filtered_signal: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: Iterable[str],
    fs: float,
    half_window: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai features de batimentos para os sinais ruidoso e filtrado."""
    if noisy_signal.shape != filtered_signal.shape:
        raise ValueError("Sinais ruidoso e filtrado precisam ter mesmo tamanho")

    samples = np.asarray(ann_samples, dtype=int)
    symbols = list(ann_symbols)

    feats_noisy = []
    feats_filtered = []
    labels = []

    for i in range(1, len(samples) - 1):
        center = samples[i]
        start = center - half_window
        end = center + half_window

        if start < 0 or end > noisy_signal.size:
            continue

        noisy_segment = noisy_signal[start:end]
        filtered_segment = filtered_signal[start:end]

        pre_rr = (samples[i] - samples[i - 1]) / fs
        post_rr = (samples[i + 1] - samples[i]) / fs

        feats_noisy.append(build_features_from_segment(noisy_segment, pre_rr, post_rr, noisy_signal[center]))
        feats_filtered.append(build_features_from_segment(filtered_segment, pre_rr, post_rr, filtered_signal[center]))
        labels.append(beat_to_binary(symbols[i]))

    if not labels:
        return np.empty((0, 13)), np.empty((0, 13)), np.empty((0,), dtype=int)

    return np.vstack(feats_noisy), np.vstack(feats_filtered), np.asarray(labels, dtype=int)


def _require_wfdb() -> None:
    if wfdb is None:
        raise RuntimeError(
            "A biblioteca 'wfdb' é necessária para carregar os arquivos locais MIT-BIH/NSTDB. "
            "Instale com: pip install wfdb"
        ) from WFDB_IMPORT_ERROR


def _resolve_record_base_path(data_dir: str, database: str, record: str) -> str:
    base_dir = Path(data_dir)
    candidates = [
        base_dir / database / record,
        base_dir / record,
    ]

    for candidate in candidates:
        if candidate.with_suffix(".hea").exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Registro '{record}' não encontrado em '{base_dir / database}' nem em '{base_dir}'. "
        f"Esperado arquivo .hea local."
    )


def load_physionet_record(data_dir: str, database: str, record: str, channel: int = 0) -> tuple[np.ndarray, float]:
    _require_wfdb()
    record_path = _resolve_record_base_path(data_dir=data_dir, database=database, record=record)
    signal, fields = wfdb.rdsamp(record_name=record_path)
    return signal[:, channel].astype(float), float(fields["fs"])


def load_mitbih_annotations(data_dir: str, record: str) -> tuple[np.ndarray, list[str]]:
    _require_wfdb()
    record_path = _resolve_record_base_path(data_dir=data_dir, database="mitdb", record=record)
    annotation = wfdb.rdann(record_name=record_path, extension="atr")
    return np.asarray(annotation.sample, dtype=int), list(annotation.symbol)


def evaluate_models(
    x_noisy: np.ndarray,
    x_filtered: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> dict[str, dict[str, float]]:
    if y.size == 0:
        raise RuntimeError("Sem batimentos válidos para treinar/classificar")

    indices = np.arange(y.size)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    metrics: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        model.fit(x_noisy[train_idx], y[train_idx])
        pred_noisy = model.predict(x_noisy[test_idx])

        model.fit(x_filtered[train_idx], y[train_idx])
        pred_filtered = model.predict(x_filtered[test_idx])

        metrics[model_name] = {
            "accuracy_noisy": float(accuracy_score(y[test_idx], pred_noisy)),
            "f1_noisy": float(f1_score(y[test_idx], pred_noisy, zero_division=0)),
            "accuracy_filtered": float(accuracy_score(y[test_idx], pred_filtered)),
            "f1_filtered": float(f1_score(y[test_idx], pred_filtered, zero_division=0)),
        }

    return metrics


def run_pipeline(records: list[str], noise_records: list[str], target_snr_db: float, seed: int, data_dir: str) -> dict:
    x_noisy_all = []
    x_filtered_all = []
    y_all = []
    snr_results: list[RecordEvaluation] = []

    for idx, record in enumerate(records):
        noise_record = noise_records[idx % len(noise_records)]

        clean, fs = load_physionet_record(data_dir=data_dir, database="mitdb", record=record)
        noise, _ = load_physionet_record(data_dir=data_dir, database="nstdb", record=noise_record)

        noisy = add_noise_at_target_snr(clean, noise, target_snr_db)
        filtered = kalman_filter_ecg(noisy)

        ann_samples, ann_symbols = load_mitbih_annotations(data_dir=data_dir, record=record)

        feats_noisy, feats_filtered, labels = extract_paired_features(
            noisy_signal=noisy,
            filtered_signal=filtered,
            ann_samples=ann_samples,
            ann_symbols=ann_symbols,
            fs=fs,
        )

        if labels.size == 0:
            continue

        x_noisy_all.append(feats_noisy)
        x_filtered_all.append(feats_filtered)
        y_all.append(labels)

        snr_results.append(
            RecordEvaluation(
                record=record,
                snr_noisy_db=compute_snr_db(clean[: noisy.size], noisy),
                snr_filtered_db=compute_snr_db(clean[: filtered.size], filtered),
            )
        )

    if not y_all:
        raise RuntimeError("Nenhum batimento válido foi extraído. Verifique os registros usados.")

    x_noisy = np.vstack(x_noisy_all)
    x_filtered = np.vstack(x_filtered_all)
    y = np.concatenate(y_all)

    class_metrics = evaluate_models(x_noisy=x_noisy, x_filtered=x_filtered, y=y, random_state=seed)

    snr_noisy_avg = float(np.mean([item.snr_noisy_db for item in snr_results]))
    snr_filtered_avg = float(np.mean([item.snr_filtered_db for item in snr_results]))

    return {
        "config": {
            "records": records,
            "noise_records": noise_records,
            "target_snr_db": target_snr_db,
            "seed": seed,
            "binary_classes": {"0": "normal", "1": "arrhythmia"},
        },
        "snr": {
            "per_record": [item.__dict__ for item in snr_results],
            "average_noisy_db": snr_noisy_avg,
            "average_filtered_db": snr_filtered_avg,
            "average_gain_db": snr_filtered_avg - snr_noisy_avg,
        },
        "classification": class_metrics,
        "dataset": {
            "n_beats": int(y.size),
            "n_arrhythmia": int(np.sum(y == 1)),
            "n_normal": int(np.sum(y == 0)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="118,119", help="Registros MIT-BIH separados por vírgula")
    parser.add_argument("--noise-records", default="bw,ma,em", help="Registros NSTDB separados por vírgula")
    parser.add_argument("--target-snr-db", type=float, default=6.0, help="SNR alvo para contaminação do ECG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data", help="Diretório base local contendo mitdb/nstdb")
    parser.add_argument("--output-json", default="", help="Caminho opcional para salvar métricas em JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = [rec.strip() for rec in args.records.split(",") if rec.strip()]
    noise_records = [rec.strip() for rec in args.noise_records.split(",") if rec.strip()]
    if not records or not noise_records:
        raise ValueError("É necessário informar ao menos 1 registro MIT-BIH e 1 registro de ruído")

    results = run_pipeline(
        records=records,
        noise_records=noise_records,
        target_snr_db=args.target_snr_db,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    output = json.dumps(results, indent=2, ensure_ascii=False)
    print(output)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fp:
            fp.write(output)


if __name__ == "__main__":
    main()
