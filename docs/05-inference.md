# Inferencia

Cómo usar los modelos entrenados (`models/runs/<Problema>/<score>__<experimento>/<score>__<experimento>.<ext>`).
Ambos scripts aceptan redes (`.pkl`) y modelos clásicos (`.joblib`), uno por problema o una
sola red conjunta, que predice los dos.

## CSV de la evaluación

[`src/predict.py`](../src/predict.py) recibe el CSV tokenizado de la evaluación
(`instancia_id,token_id,token`), predice cada instancia como una secuencia y escribe una fila
por token:

```
instancia_id,token_id,token,punt_inicial,punt_final,capitalización
1,10854,cu,¿,,1
1,101439,##ánd,,,1
1,10133,##o,,,1
...
1,187,s,,?,2
```

Igual que en la métrica: la capitalización de la palabra (la más votada) va en todos sus
tokens, el `¿` en el primero y el `,` `.` `?` en el último. Se usa un modelo por problema o
una sola red conjunta:

```bash
uv run python -m src.predict eval.csv -o predicciones.csv -m <modelo de capitalización> -m <modelo de puntuación>
```

## Prueba interactiva

[`src/inference.py`](../src/inference.py) pide un texto, lo normaliza como el dataset (sin
signos y en minúsculas), corre los modelos y reconstruye el texto con la capitalización y la
puntuación predichas. Sin argumentos abre un menú para elegir la corrida y el modelo de cada
problema:

```bash
uv run python -m src.inference
uv run python -m src.inference -m <modelo de capitalización> -m <modelo de puntuación> -v
```

Con `-v` (o desde el menú) muestra además la clase predicha de cada palabra. La capitalización
MiXta no dice qué letras van en mayúscula, así que esas palabras se marcan con `*`.
