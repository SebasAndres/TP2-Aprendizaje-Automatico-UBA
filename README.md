# tp aa2

> Ver [PENDIENTES](PENDIENTES.md) y [Entrenar en Google Colab](docs/google-colab-training.md) 👀

Para ejecutar se usa
~~~bash
uv run python -m src.pipeline
~~~

Antes de la primera ejecucion ejecutar para sincronizar el entorno
~~~bash
uv sync
~~~

## RNN conjunta (prototipo)

Una sola RNN que predice capitalización, puntuación inicial y puntuación final al mismo tiempo,
compartiendo los pesos entre las tres tareas. Está fuera del pipeline para discutir si conviene
compartir pesos. Usa los mismos splits, semilla e hiperparámetros que la RNN estándar de capitalización.

~~~bash
uv run python -m src.joint_rnn --sentences 30000                  # RNN estándar, ~1 min en CPU
uv run python -m src.joint_rnn --sentences 30000 --bidirectional  # RNN bidireccional, ~1 min en CPU
uv run python -m src.joint_rnn                                    # dataset completo
~~~
