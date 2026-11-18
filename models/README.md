# models/

Todos los modelos entrenados, uno por carpeta, y lo que compara varios modelos.

```
models/
├── runs/<Problema>/<score>__<modelo>/   ← un modelo
│   ├── <score>__<modelo>.pkl            ← pesos (.pkl = red, .joblib = clásico)
│   ├── metadata.json                    ← corrida, config, tamaños de los splits y métricas
│   ├── train_test.png                   ← curvas por época (los clásicos: test.png, por clase)
│   ├── test_confusion.png               ← matrices de confusión sobre test (corridas nuevas)
│   ├── control.png                      ← precision/recall por clase sobre control
│   └── control_confusion.png            ← matrices de confusión sobre control
├── reports/<nombre>/                    ← comparaciones entre modelos
│   ├── final/                           ← póster y metrics.json del experimento final
│   ├── precandidato_final/              ← ídem del candidato anterior (10 épocas)
│   └── <día>__<hora>/                   ← pósters y matrices de confusión de corridas viejas
├── learning_rate/                       ← búsqueda de learning rate (results.json)
└── regularization/                      ← búsqueda de weight decay / dropout (results.json)
```

- `<score>` es el F1 macro en **test** (promedio de las tareas del problema). Para comparar
  modelos usar **control**, que está en `metadata.json`.
- `metadata.json` tiene `run` (la corrida `<día>__<hora>` del pipeline), `config`, `data`, `test`,
  `control`, `best_epoch` e `history` (ver `src/training/report.py`).
- `src.pipeline` guarda acá cada modelo nuevo. Si ya hay uno con el mismo `<score>__<modelo>`,
  se le agrega la corrida al nombre de la carpeta.
- `src.inference` lista estos modelos y `src.predict -m <carpeta>/<score>__<modelo>.pkl` los usa.
- Para juntar los modelos de una o varias corridas en un reporte:
  `src.training.report.run_report("<día>__<hora>")` o `candidato_final_plots.merge_runs([...])`.
- Las redes conjuntas (`joint_*`) se guardan una vez por problema, con el score de las tareas de
  ese problema. Los dos `.pkl` son la misma red.
- `models/` está en `.gitignore`: para subir algo hay que usar `git add -f`.

## Modelos finales

Experimento final (`reports/final/`, armado por `notebooks/experiments/final_resultados.ipynb`).
Las predicciones de `competencia/` salen de la Conv + BiLSTM conjunta.

| Problema | Modelo | Carpeta | F1 control | F1 test | Corrida |
|---|---|---|---|---|---|
| Capitalización | Conv + BiLSTM conjunta, 20 épocas | `runs/CapitalizationProblem/0.774__joint_conv_bilstm` | 0.788 | 0.774 | 2026-11-18__04-38-20 |
| Puntuación | Conv + BiLSTM conjunta, 20 épocas | `runs/PunctuationProblem/0.792__joint_conv_bilstm` | 0.787 | 0.792 | 2026-11-18__04-38-20 |
| Capitalización | BiRNN, 15 épocas, dropout 0.3, cabeza 128 | `runs/CapitalizationProblem/0.706__bidirectional_rnn` | 0.728 | 0.706 | 2026-11-18__04-44-00 |
| Puntuación | BiRNN, 15 épocas, dropout 0.3, cabeza 128, wd 0.1 | `runs/PunctuationProblem/0.750__bidirectional_rnn` | 0.745 | 0.750 | 2026-11-18__04-44-00 |

El candidato anterior (`reports/precandidato_final/`, 10 épocas) son las corridas
`2026-11-18__02-41-48` (Conv + BiLSTM) y `2026-11-18__01-46-37` (BiRNN).

Los modelos de las corridas `matias__*` los subió Matias por separado: no tienen métricas ni config.

## Índice

"En git" = la carpeta está subida al repo. Las de las corridas viejas quedaron solo en local.

### CapitalizationProblem

| Carpeta | Corrida | F1 test | F1 control | Épocas (mejor) | En git |
|---|---|---|---|---|---|
| `0.774__joint_conv_bilstm` | 2026-11-18__04-38-20 | 0.774 | 0.788 | 20 (18) | sí |
| `0.744__joint_conv_bilstm` | 2026-11-18__02-41-48 | 0.744 | 0.758 | 10 (10) | sí |
| `0.706__bidirectional_rnn` | 2026-11-18__04-44-00 | 0.706 | 0.728 | 15 (15) | sí |
| `0.675__bidirectional_rnn` | 2026-11-18__01-46-37 | 0.675 | 0.698 | 10 (9) | sí |
| `0.671__bidirectional_rnn` | matias__capitalizacion | - | - | - | sí |
| `0.604__joint_conv_bilstm` | 2026-11-16__13-37-08 | 0.604 | 0.616 | 15 (15) | no |
| `0.587__classic_features_dictionary` | 2026-11-16__13-37-08 | 0.587 | 0.634 | - | no |
| `0.585__joint_conv_bilstm` | 2026-11-16__21-24-10 | 0.585 | 0.614 | 5 (4) | sí |
| `0.583__bidirectional_rnn` | 2026-11-16__21-24-10 | 0.583 | 0.605 | 5 (5) | sí |
| `0.572__classic_features_dictionary` | 2026-11-16__00-20-22 | 0.572 | 0.549 | - | no |
| `0.554__joint_residual_stacked_bilstm` | 2026-11-16__13-37-08 | 0.554 | 0.563 | 15 (15) | no |
| `0.545__joint_projected_bigru` | 2026-11-16__13-37-08 | 0.545 | 0.558 | 15 (15) | no |
| `0.541__bidirectional_joint_lstm` | 2026-11-16__13-37-08 | 0.541 | 0.543 | 15 (14) | no |
| `0.530__classic_embeddings` | 2026-11-16__00-20-22 | 0.530 | 0.538 | - | no |
| `0.524__bidirectional_rnn` | 2026-11-16__13-37-08 | 0.524 | 0.530 | 15 (11) | no |
| `0.524__bidirectional_joint_rnn` | 2026-11-16__13-37-08 | 0.524 | 0.546 | 15 (12) | no |
| `0.496__classic_embeddings` | 2026-11-16__13-37-08 | 0.496 | 0.565 | - | no |
| `0.492__joint_rnn` | 2026-11-16__13-37-08 | 0.492 | 0.521 | 15 (15) | no |
| `0.488__joint_lstm` | 2026-11-16__13-37-08 | 0.488 | 0.512 | 15 (15) | no |
| `0.479__standard_rnn` | 2026-11-16__13-37-08 | 0.479 | 0.482 | 15 (9) | no |
| `0.479__bidirectional_rnn` | 2026-11-16__00-20-22 | 0.479 | 0.427 | - | no |
| `0.440__standard_rnn` | 2026-11-16__00-20-22 | 0.440 | 0.406 | - | no |
| `0.433__bidirectional_joint_rnn` | 2026-11-16__00-20-22 | 0.433 | 0.427 | - | no |
| `0.406__joint_rnn` | 2026-11-16__00-20-22 | 0.406 | 0.363 | - | no |
| `0.402__bidirectional_joint_lstm` | 2026-11-16__00-20-22 | 0.402 | 0.372 | - | no |
| `0.356__classic_features` | 2026-11-16__00-20-22 | 0.356 | 0.340 | - | no |
| `0.346__classic_features` | 2026-11-16__13-37-08 | 0.346 | 0.395 | - | no |
| `0.342__joint_lstm` | 2026-11-16__00-20-22 | 0.342 | 0.343 | - | no |
| `0.336__classic_features_dictionary` | 2026-11-15__23-43-41 | 0.336 | 0.348 | - | no |
| `0.326__bidirectional_joint_rnn` | 2026-11-15__23-31-41 | - | - | - | no |
| `0.316__classic_embeddings` | 2026-11-15__23-43-41 | 0.316 | 0.330 | - | no |
| `0.295__classic_features` | 2026-11-15__23-43-41 | 0.295 | 0.295 | - | no |
| `0.277__joint_lstm` | 2026-11-15__23-43-41 | 0.277 | 0.268 | 5 (2) | no |
| `0.240__bidirectional_rnn` | 2026-11-15__23-43-41 | 0.240 | 0.231 | 5 (5) | no |
| `0.222__standard_rnn` | 2026-11-15__23-43-41 | 0.222 | 0.221 | 5 (1) | no |
| `0.222__bidirectional_joint_rnn` | 2026-11-15__23-43-41 | 0.222 | 0.221 | 5 (1) | no |
| `0.222__bidirectional_joint_lstm` | 2026-11-15__23-43-41 | 0.222 | 0.221 | 5 (1) | no |
| `0.221__joint_rnn` | 2026-11-15__23-43-41 | 0.221 | 0.221 | 5 (1) | no |

### PunctuationProblem

| Carpeta | Corrida | F1 test | F1 control | Épocas (mejor) | En git |
|---|---|---|---|---|---|
| `0.792__joint_conv_bilstm` | 2026-11-18__04-38-20 | 0.792 | 0.787 | 20 (18) | sí |
| `0.777__joint_conv_bilstm` | 2026-11-18__02-41-48 | 0.777 | 0.772 | 10 (10) | sí |
| `0.750__bidirectional_rnn` | 2026-11-18__04-44-00 | 0.750 | 0.745 | 15 (14) | sí |
| `0.748__bidirectional_rnn` | matias__punctuation_birr_puntini | - | - | - | sí |
| `0.729__bidirectional_rnn` | 2026-11-18__01-46-37 | 0.729 | 0.723 | 10 (10) | sí |
| `0.673__joint_residual_stacked_bilstm` | 2026-11-16__13-37-08 | 0.673 | 0.676 | 15 (15) | no |
| `0.671__bidirectional_rnn` | 2026-11-16__21-24-10 | 0.671 | 0.667 | 5 (4) | sí |
| `0.666__joint_projected_bigru` | 2026-11-16__13-37-08 | 0.666 | 0.671 | 15 (15) | no |
| `0.664__joint_conv_bilstm` | 2026-11-16__13-37-08 | 0.664 | 0.669 | 15 (15) | no |
| `0.660__bidirectional_joint_lstm` | 2026-11-16__13-37-08 | 0.660 | 0.664 | 15 (14) | no |
| `0.657__joint_conv_bilstm` | 2026-11-16__21-24-10 | 0.657 | 0.653 | 5 (4) | sí |
| `0.655__bidirectional_joint_rnn` | 2026-11-16__13-37-08 | 0.655 | 0.659 | 15 (12) | no |
| `0.649__bidirectional_rnn` | 2026-11-16__13-37-08 | 0.649 | 0.651 | 15 (15) | no |
| `0.607__bidirectional_rnn` | 2026-11-16__00-20-22 | 0.607 | 0.617 | - | no |
| `0.597__bidirectional_joint_rnn` | 2026-11-16__00-20-22 | 0.597 | 0.607 | - | no |
| `0.582__joint_lstm` | 2026-11-16__13-37-08 | 0.582 | 0.587 | 15 (15) | no |
| `0.579__standard_rnn` | 2026-11-16__13-37-08 | 0.579 | 0.582 | 15 (12) | no |
| `0.577__joint_rnn` | 2026-11-16__13-37-08 | 0.577 | 0.580 | 15 (15) | no |
| `0.567__bidirectional_joint_lstm` | 2026-11-16__00-20-22 | 0.567 | 0.573 | - | no |
| `0.563__classic_features_dictionary` | 2026-11-16__00-20-22 | 0.563 | 0.570 | - | no |
| `0.560__classic_features_dictionary` | 2026-11-16__13-37-08 | 0.560 | 0.561 | - | no |
| `0.544__standard_rnn` | 2026-11-16__00-20-22 | 0.544 | 0.545 | - | no |
| `0.541__joint_rnn` | 2026-11-16__00-20-22 | 0.541 | 0.544 | - | no |
| `0.535__classic_features` | 2026-11-16__13-37-08 | 0.535 | 0.534 | - | no |
| `0.533__classic_features` | 2026-11-16__00-20-22 | 0.533 | 0.538 | - | no |
| `0.517__bidirectional_joint_rnn` | 2026-11-15__23-31-41 | - | - | - | no |
| `0.515__classic_embeddings` | 2026-11-16__13-37-08 | 0.515 | 0.515 | - | no |
| `0.506__joint_lstm` | 2026-11-16__00-20-22 | 0.506 | 0.516 | - | no |
| `0.480__classic_features_dictionary` | 2026-11-15__23-43-41 | 0.480 | 0.506 | - | no |
| `0.480__classic_embeddings` | 2026-11-15__23-43-41 | 0.480 | 0.523 | - | no |
| `0.455__classic_features` | 2026-11-15__23-43-41 | 0.455 | 0.484 | - | no |
| `0.423__classic_embeddings` | 2026-11-16__00-20-22 | 0.423 | 0.421 | - | no |
| `0.359__joint_lstm` | 2026-11-15__23-43-41 | 0.359 | 0.354 | 5 (2) | no |
| `0.359__bidirectional_joint_rnn` | 2026-11-15__23-43-41 | 0.359 | 0.354 | 5 (1) | no |
| `0.358__joint_rnn` | 2026-11-15__23-43-41 | 0.358 | 0.355 | 5 (1) | no |
| `0.357__standard_rnn` | 2026-11-15__23-43-41 | 0.357 | 0.354 | 5 (2) | no |
| `0.357__bidirectional_rnn` | 2026-11-15__23-43-41 | 0.357 | 0.354 | 5 (2) | no |
| `0.357__bidirectional_joint_lstm` | 2026-11-15__23-43-41 | 0.357 | 0.354 | 5 (1) | no |
