# Setup

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
    -m models/runs/CapitalizationProblem/0.706__bidirectional_rnn/0.706__bidirectional_rnn.pkl \
    -m models/runs/PunctuationProblem/0.750__bidirectional_rnn/0.750__bidirectional_rnn.pkl
~~~
