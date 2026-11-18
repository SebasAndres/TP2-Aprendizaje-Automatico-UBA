# Métricas y loss

Con qué se entrenan las redes (la loss) y con qué se evalúan y comparan todos los modelos (la
métrica). Las redes y su entrenamiento están en [03-models.md](03-models.md).

## Loss

Solo las redes la usan para entrenar; los RandomForest eligen sus cortes por impureza de
Gini (ver [03-models.md](03-models.md)). Está en [`src/metrics/base.py`](../src/metrics/base.py):

| Problema | Loss |
|---|---|
| Capitalización | `CrossEntropy(logits, y)` por token |
| Puntuación | `CrossEntropy(inicial) + CrossEntropy(final)` |
| Conjunta | `CrossEntropy(capitalización) + CrossEntropy(inicial) + CrossEntropy(final)` |

La CrossEntropy de la tarea $k$ en un batch, sobre los tokens $t$ que no son padding, con
$y_t$ la clase real y $w_c$ el peso de la clase $c$:

$$\mathcal{L}_k = -\frac{\sum_t w_{y_t} \log \mathrm{softmax}\big(z^{(k)}_t\big)_{y_t}}{\sum_t w_{y_t}}$$

(es el promedio pesado que calcula `F.cross_entropy` con `weight`). La loss de una red es la
suma sobre sus tareas: $\mathcal{L} = \sum_k \mathcal{L}_k$.

- El padding (`-100`) se ignora (`ignore_index`).
- **Pesos por clase** ([`src/training/weights.py`](../src/training/weights.py)): la mayoría de
  los tokens no lleva signo ni mayúscula, así que equivocarse en una clase rara cuesta más. El
  peso de la clase $c$ de cada tarea es $w_c = \left(\frac{N}{K \cdot n_c}\right)^{p}$ con
  $p$ = `CLASS_WEIGHT_POWER`, calculado sobre
  train, con `N` los tokens de la tarea, `K` las clases presentes y `n_c` los tokens de la
  clase. Con `CLASS_WEIGHT_POWER = 0.5` (default) es la raíz del balanceo completo.
- La loss es **por token**, aunque la métrica sea por palabra: en capitalización todos los
  tokens de la palabra tienen la etiqueta; en puntuación, los tokens del medio de una palabra
  aprenden "nada".

## Métrica

La del enunciado: **F1 macro a nivel palabra, por tarea** ([`src/metrics/`](../src/metrics/)).
Aunque los modelos predicen por token, cada palabra cuenta una vez, sin importar en cuántos
tokens se parta.

| Tarea | Clase real de la palabra | Clase predicha de la palabra | F1 macro sobre |
|---|---|---|---|
| `capitalizacion` | la del primer token (todos tienen la misma) | la más votada entre sus tokens (empate: la que aparece primero) | 4 clases |
| `punt_inicial` | la del primer token | la del primer token | 2 clases |
| `punt_final` | la del último token | la del último token | 4 clases |

Con $\hat{y}_t$ la clase predicha de cada token, la predicha de la palabra $w$ en
capitalización es la más votada:

$$\hat{y}_w = \arg\max_c \big|\{\, t \in w : \hat{y}_t = c \,\}\big|$$

Sobre las palabras, para cada clase $c$ de la tarea, con $TP_c$, $FP_c$ y $FN_c$ los
verdaderos positivos, falsos positivos y falsos negativos:

$$P_c = \frac{TP_c}{TP_c + FP_c}, \qquad R_c = \frac{TP_c}{TP_c + FN_c}, \qquad F_{1,c} = \frac{2\, P_c R_c}{P_c + R_c}, \qquad F_1^{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F_{1,c}$$

(una división por cero vale 0).

- El macro promedia **todas** las clases, aparezcan o no en los datos (una clase ausente cuenta
  como F1 = 0). Como pesa lo mismo cada clase, las raras (MiXta, MAYÚS, `¿`, `?`) mueven mucho
  el resultado aunque sean pocas palabras.
- Para comparar modelos de un problema se usa el **promedio del F1 macro de sus tareas** $\mathcal{T}$,
  $\overline{F_1} = \frac{1}{|\mathcal{T}|} \sum_{k \in \mathcal{T}} F_1^{\text{macro}}(k)$
  (`mean_score`): para elegir la época de cada red, el modelo ganador y el score del nombre del
  archivo.
- Cada modelo guarda en `models/runs/<Problema>/<score>__<experimento>/metadata.json` su corrida,
  la config, el F1 macro y la matriz de confusión por tarea, en test y en control, y el historial
  por época de las redes (ver [`src/training/report.py`](../src/training/report.py)).
  `run_report` junta los de una corrida en el reporte que lee `notebooks/poster.ipynb` para los
  gráficos.
