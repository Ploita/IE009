# IE009

Pipeline em Python para:
- adicionar ruído de NSTDB em sinais de ECG da MIT-BIH Arrhythmia,
- filtrar o ruído com Filtro de Kalman,
- avaliar ganho de SNR,
- comparar classificação binária de batimentos (normal vs arritmia) com sinais ruidosos e filtrados.

## Requisitos

```bash
pip install numpy scikit-learn wfdb
```

## Execução com dados do PhysioNet (MIT-BIH + NSTDB)

```bash
python ecg_kalman_pipeline.py \
  --records 100,101,103 \
  --noise-records bw,ma,em \
  --target-snr-db 6 \
  --output-json resultados.json
```

Saída principal:
- `snr.average_noisy_db`
- `snr.average_filtered_db`
- `snr.average_gain_db`
- métricas de classificação (`accuracy` e `f1`) para modelos tradicionais (`logistic_regression` e `random_forest`) em sinais ruidosos e filtrados.

## Smoke test local (sem download)

```bash
python ecg_kalman_pipeline.py --synthetic-smoke
```
