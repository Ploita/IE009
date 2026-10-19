# IE009

Pipeline em Python para:
- adicionar ruído de NSTDB em sinais de ECG da MIT-BIH Arrhythmia local,
- filtrar o ruído com Filtro de Kalman,
- avaliar ganho de SNR,
- comparar classificação binária de batimentos (normal vs arritmia) com sinais ruidosos e filtrados.

## Requisitos

```bash
pip install numpy scikit-learn wfdb
```

## Organização esperada dos dados

O script usa apenas dados locais em `./data` (sem download).  
Ele procura os registros em:
- `./data/mitdb/<registro>.{hea,dat,atr}` (ou `./data/<registro>.{hea,dat,atr}`)
- `./data/nstdb/<registro>.{hea,dat}` (ou `./data/<registro>.{hea,dat}`)

## Execução com base local (MIT-BIH + NSTDB)

```bash
python ecg_kalman_pipeline.py \
  --data-dir ./data \
  --records 118,119 \
  --noise-records bw,ma,em \
  --target-snr-db 6 \
  --output-json resultados.json
```

Saída principal:
- `snr.average_noisy_db`
- `snr.average_filtered_db`
- `snr.average_gain_db`
- métricas de classificação (`accuracy` e `f1`) para modelos tradicionais (`logistic_regression` e `random_forest`) em sinais ruidosos e filtrados.
