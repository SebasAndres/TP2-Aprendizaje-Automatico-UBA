# Predicción de puntuación y capitalización

Dado un texto en minúsculas y sin puntuación, tokenizado con `bert-base-multilingual-cased`, se
predice para cada token su capitalización y su puntuación inicial y final (ver `enunciado.pdf`).
El pipeline (`uv run python -m src.pipeline`) entrena y compara todos los modelos de cada
problema.

## Documentos

| Documento | Qué tiene |
|---|---|
| [intro.md](intro.md) | Este resumen: problemas, tareas y experimentos |
| [datasets.md](datasets.md) | De texto a instancias, qué es una palabra y los splits |
| [features_and_expected_output.md](features_and_expected_output.md) | Feature extractors y etiquetas esperadas de cada problema |
| [models.md](models.md) | Modelos clásicos y redes: entrada, salida, loss, entrenamiento y métrica |
| [inference.md](inference.md) | CSV de predicciones de la evaluación y prueba interactiva |
| [google-colab-training.md](google-colab-training.md) | Cómo correr el pipeline en Colab |

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

Los modelos se guardan en `models/<día>__<hora>/<Problema>/<score>__<experimento>.<ext>`, con
el score como el promedio del F1 macro en test de las tareas del problema.

Cómo se arman las instancias: [datasets.md](datasets.md). Qué recibe y qué debe devolver cada
modelo: [features_and_expected_output.md](features_and_expected_output.md) y
[models.md](models.md).
