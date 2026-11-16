# tp aa2

Antes de la primera ejecucion ejecutar para sincronizar el entorno
~~~bash
uv sync
~~~

Para ejecutar el entrenamiento se usa
~~~bash
uv run python -m src.pipeline
~~~

Para iterar rápido con pocos datos y solo algunos experimentos (los nombres de `build_experiments`
en `src/training/train.py`):
~~~bash
uv run python -m src.pipeline --experiments standard_rnn,joint_lstm \
    --capitalization-size 300 --punctuation-size 300
~~~

Para probar RT los modelos se usa
~~~bash
uv run python -m src.inference
~~~

Para generar el `.csv` de predicciones del enunciado (entrada: `instancia_id,token_id,token`)
se pasa un modelo por problema, o una sola red conjunta:
~~~bash
uv run python -m src.predict eval.csv -o predicciones.csv \
    -m models/<día>__<hora>/CapitalizationProblem/<score>__bidirectional_rnn.pkl \
    -m models/<día>__<hora>/PunctuationProblem/<score>__bidirectional_rnn.pkl
~~~

## RNN conjunta (prototipo)

Una sola RNN que predice capitalización, puntuación inicial y puntuación final al mismo tiempo,
compartiendo los pesos entre las tres tareas. Está fuera del pipeline para discutir si conviene
compartir pesos. Usa los mismos splits, semilla e hiperparámetros que la RNN estándar de capitalización.

~~~bash
uv run python -m src.joint_rnn --size 10000                  # RNN estándar, hasta 10000 instancias por fuente
uv run python -m src.joint_rnn --size 10000 --bidirectional  # RNN bidireccional
uv run python -m src.joint_rnn                               # todas las instancias de cada fuente
uv run python -m src.joint_rnn --bidirectional --lstm        # LSTM bidireccional en lugar de RNN simple
~~~
