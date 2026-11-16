# Predicción de puntuación y capitalización

Dado un texto en minúsculas y sin puntuación, tokenizado con `bert-base-multilingual-cased`, se
predice para cada token su capitalización y su puntuación inicial y final (ver `enunciado.pdf`).
El pipeline (`uv run python -m src.pipeline`) entrena y compara todos los modelos de cada
problema.

## Documentos

| Documento | Qué tiene |
|---|---|
| [00-intro.md](00-intro.md) | Este resumen: problemas, tareas y experimentos |
| [01-datasets.md](01-datasets.md) | De texto a instancias, qué es una palabra y los splits |
| [02-features_and_expected_output.md](02-features_and_expected_output.md) | Feature extractors y etiquetas esperadas de cada problema |
| [03-models.md](03-models.md) | Modelos clásicos y redes: entrada, salida y entrenamiento |
| [04-metrics.md](04-metrics.md) | Loss de las redes y métrica con la que se comparan los modelos |
| [05-inference.md](05-inference.md) | CSV de predicciones de la evaluación y prueba interactiva |
| [google-colab-setup.md](google-colab-setup.md) | Cómo correr el pipeline en Colab |

## Resumen

Hay **2 problemas** y **3 tareas**:

| Problema | Tareas | Clases |
|---|---|---|
| Capitalización (`CapitalizationProblem`) | `capitalizacion` | 0 = minúscula, 1 = Inicial, 2 = MiXta, 3 = MAYÚS |
| Puntuación (`PunctuationProblem`) | `punt_inicial` | 0 = nada, 1 = `¿` |
| | `punt_final` | 0 = nada, 1 = `,`, 2 = `.`, 3 = `?` |

Cada experimento de [`build_experiments`](../src/training/train.py) se entrena **una vez por
problema**:

| Experimento | Modelo | Features | Nivel | Archivo |
|---|---|---|---|---|
| `classic_features` | RandomForest | `ClassicFeatureExtractor` (48 atributos) | palabra | `.joblib` |
| `classic_features_dictionary` | RandomForest | `ClassicFeatureExtractor` + diccionario (54) | palabra | `.joblib` |
| `classic_embeddings` | RandomForest | `AverageEmbeddingsExtractor` (768) | palabra | `.joblib` |
| `standard_rnn` | RNN | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `bidirectional_rnn` | RNN bidireccional | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `joint_rnn` | RNN conjunta (3 tareas) | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `bidirectional_joint_rnn` | RNN bidireccional conjunta | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `joint_lstm` | LSTM conjunta | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `bidirectional_joint_lstm` | LSTM bidireccional conjunta | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `joint_projected_bigru` | Proyección por token + GRU bidireccional conjunta | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `joint_residual_stacked_bilstm` | LSTM bidireccional de 2 capas conjunta, con residual | `EmbeddingsExtractor` (768) | token | `.pkl` |
| `joint_conv_bilstm` | Convolución + LSTM bidireccional conjunta | `EmbeddingsExtractor` (768) | token | `.pkl` |

Los modelos se guardan en `models/<día>__<hora>/<Problema>/<score>__<experimento>.<ext>`, con
el score como el promedio del F1 macro en test de las tareas del problema.

Cómo se arman las instancias: [01-datasets.md](01-datasets.md). Qué recibe y qué debe devolver cada
modelo: [02-features_and_expected_output.md](02-features_and_expected_output.md) y
[03-models.md](03-models.md). Cómo se comparan: [04-metrics.md](04-metrics.md).
