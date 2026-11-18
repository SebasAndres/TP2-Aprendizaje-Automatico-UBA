# models/

Modelos entrenados (`.pkl` = redes, `.joblib` = clásicos) y su metadata (`metrics.json`, gráficos).

```
models/
├── final/                 ← experimento final: los modelos a usar
├── precandidato_final/    ← corrida anterior al final (10 épocas), con la que se compara
├── corridas/              ← corridas crudas del pipeline, models/corridas/<día>__<hora>/
├── learning_rate/         ← búsqueda de learning rate (solo results.json)
├── regularization/        ← búsqueda de weight decay / dropout (solo results.json)
└── punctuation_birr_puntini/
```

Las carpetas `final/` y `precandidato_final/` tienen el mismo formato que una corrida
(`<Problema>/<score>__<modelo>.pkl`), así que se pasan directo a `src/predict.py` y aparecen en
el menú de `src/inference.py`. Cada una junta dos corridas (una por modelo). Los `.pkl` son copias
de esas corridas, que siguen en `corridas/` con sus gráficos de entrenamiento y matrices de
confusión. `metrics.json` → `config.merged_from` dice de qué corridas sale.

El score del nombre del archivo es el F1 macro en **test** (promedio de las tareas del problema).

## final/

Armada por `notebooks/experiments/final_resultados.ipynb`. Póster en `final/poster/`.

| Problema | Modelo | `.pkl` | F1 control | F1 test | Épocas (mejor) | Corrida de origen | Notebook que la entrenó |
|---|---|---|---|---|---|---|---|
| Capitalización | Conv + BiLSTM conjunta | `CapitalizationProblem/0.774__joint_conv_bilstm.pkl` | 0.788 | 0.774 | 20 (18) | `corridas/2026-11-18__04-38-20` | `final_joint_20_epocas.ipynb` |
| Puntuación | Conv + BiLSTM conjunta | `PunctuationProblem/0.792__joint_conv_bilstm.pkl` | 0.787 | 0.792 | 20 (18) | `corridas/2026-11-18__04-38-20` | `final_joint_20_epocas.ipynb` |
| Capitalización | BiRNN (dropout 0.3, cabeza 128) | `CapitalizationProblem/0.706__bidirectional_rnn.pkl` | 0.728 | 0.706 | 15 (15) | `corridas/2026-11-18__04-44-00` | `final_birnn_dropout_cabeza.ipynb` |
| Puntuación | BiRNN (dropout 0.3, cabeza 128, wd 0.1) | `PunctuationProblem/0.750__bidirectional_rnn.pkl` | 0.745 | 0.750 | 15 (14) | `corridas/2026-11-18__04-44-00` | `final_birnn_dropout_cabeza.ipynb` |

Las predicciones de `competencia/` salen de la Conv + BiLSTM conjunta (`corridas/2026-11-18__04-38-20`).

## precandidato_final/

Armada por `notebooks/experiments/candidato_final_resultados.ipynb`. Póster en `precandidato_final/poster/`.

| Problema | Modelo | `.pkl` | F1 control | F1 test | Épocas (mejor) | Corrida de origen |
|---|---|---|---|---|---|---|
| Capitalización | Conv + BiLSTM conjunta | `CapitalizationProblem/0.744__joint_conv_bilstm.pkl` | 0.758 | 0.744 | 10 (10) | `corridas/2026-11-18__02-41-48` |
| Puntuación | Conv + BiLSTM conjunta | `PunctuationProblem/0.777__joint_conv_bilstm.pkl` | 0.772 | 0.777 | 10 (10) | `corridas/2026-11-18__02-41-48` |
| Capitalización | BiRNN | `CapitalizationProblem/0.675__bidirectional_rnn.pkl` | 0.698 | 0.675 | 10 (9) | `corridas/2026-11-18__01-46-37` |
| Puntuación | BiRNN (wd 0.1) | `PunctuationProblem/0.729__bidirectional_rnn.pkl` | 0.723 | 0.729 | 10 (10) | `corridas/2026-11-18__01-46-37` |

## corridas/

| Corrida | Qué es | En git |
|---|---|---|
| `2026-11-18__04-44-00` | BiRNN final (15 épocas, dropout + cabeza) | sí |
| `2026-11-18__04-38-20` | Conv + BiLSTM final (20 épocas) | sí |
| `2026-11-18__02-41-48` | Conv + BiLSTM del precandidato (10 épocas) | sí |
| `2026-11-18__01-46-37` | BiRNN del precandidato (10 épocas) | sí |
| `2026-11-16__21-24-10` | BiRNN y Conv + BiLSTM, 5 épocas | sí |
| `2026-11-16__13-37-08` | Las 9 redes + clásicos, 15 épocas (la usa `notebooks/poster.ipynb`) | no |
| `2026-11-16__00-20-22` | Redes + clásicos, 5 épocas | no |
| `2026-11-15__23-43-41` | Redes + clásicos, 5 épocas | no |
| `2026-11-15__23-31-41` | Prueba de la BiRNN conjunta | no |

## Corridas nuevas

El pipeline guarda cada corrida nueva en `models/<día>__<hora>/`. Cuando una sirva, moverla a
`corridas/`. Si pasa a ser la versión final, copiar sus `.pkl` a la carpeta que corresponda.
`models/` está en `.gitignore`: para subir algo hay que usar `git add -f`.
