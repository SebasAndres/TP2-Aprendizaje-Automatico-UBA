# Modelos

Los modelos de cada experimento (ver la tabla de [00-intro.md](00-intro.md)): su entrada y salida
y cómo se entrenan. La loss y la métrica están en [04-metrics.md](04-metrics.md). Las features y etiquetas que reciben
están en [02-features_and_expected_output.md](02-features_and_expected_output.md).

## Clásicos (RandomForest)

[`src/training/classic.py`](../src/training/classic.py). Un `RandomForestClassifier` de sklearn
(100 árboles, `random_state=12`) por problema, entrenado **a nivel palabra**:

| | Entrada `X` | Salida `y` |
|---|---|---|
| Capitalización | `(n_palabras, n_features)` | `(n_palabras,)` |
| Puntuación | `(n_palabras, n_features)` | `(n_palabras, 2)` = `[inicial, final]` (multi-salida) |

Las etiquetas por palabra salen de `metric.word_labels`: la capitalización del primer token, la
puntuación inicial del primero y la final del último. Al predecir, la clase de cada palabra se
copia a todos sus tokens. Se guarda un `ClassicModel` (el estimador y, si lo usa, el
diccionario).

No entrena por épocas ni tiene una loss explícita: cada árbol elige sus cortes por impureza de
Gini (el default de sklearn): en un nodo con proporción $p_c$ de la clase $c$,

$$G = 1 - \sum_c p_c^2$$

y cada corte minimiza la impureza de los hijos pesada por su tamaño. El bosque predice la clase
más votada entre sus árboles (en sklearn, la de mayor probabilidad promedio).

## Redes

Todas reciben lo mismo y devuelven **logits** (sin softmax) por token:

- **Entrada**: `forward(x, lengths)`, con `x` de forma `(batch, max_tokens, 768)` (embeddings
  con padding de ceros al final) y `lengths` la cantidad de tokens reales de cada instancia. La
  capa recurrente usa `pack_padded_sequence` para saltear el padding (importa sobre todo en la
  bidireccional, que si no empezaría la lectura hacia atrás por los ceros).
- **Estado oculto**: `HIDDEN_DIM = 128` por dirección.
- **Predicción**: `argmax` sobre la última dimensión: $\hat{y}_t = \arg\max_c z_{t,c}$.

**RNN** (`nn.RNN`). Para cada token $t$, con $x_t \in \mathbb{R}^{768}$ su embedding y
$h_0 = 0$:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \qquad h_t \in \mathbb{R}^{128}$$

Los mismos pesos se usan en todas las posiciones, y $h_t$ solo conoce los tokens $0..t$.

**Bidireccional**: una RNN lee de izquierda a derecha y otra de derecha a izquierda, y cada
token usa las dos:

$$\overrightarrow{h}_t = \tanh(\overrightarrow{W}_x x_t + \overrightarrow{W}_h \overrightarrow{h}_{t-1} + \overrightarrow{b}), \qquad \overleftarrow{h}_t = \tanh(\overleftarrow{W}_x x_t + \overleftarrow{W}_h \overleftarrow{h}_{t+1} + \overleftarrow{b}), \qquad h_t = [\overrightarrow{h}_t \,;\, \overleftarrow{h}_t] \in \mathbb{R}^{256}$$

**LSTM** (`nn.LSTM`): además de $h_t$ lleva una memoria $c_t$ que solo cambia a través de
compuertas, con $\sigma$ la sigmoide y $\odot$ el producto elemento a elemento:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{(qué escribir)} \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{(qué olvidar)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{(qué leer)} \\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

**Cabezas**: cada tarea $k$ lee el mismo $h_t$ con su propia capa lineal, que da los logits

$$z^{(k)}_t = W^{(k)} h_t + b^{(k)} \in \mathbb{R}^{C_k}$$

con $C_k$ las clases de la tarea (4, 2 o 4).

| Experimento | Clase | Capa recurrente | Cabezas | Salida | Parámetros |
|---|---|---|---|---|---|
| `standard_rnn` (capitalización) | `CapitalizationRNN` | `nn.RNN(768, 128)` | 1 lineal → 4 | `(B, T, 4)` | 115 K |
| `bidirectional_rnn` (capitalización) | `CapitalizationBidirectionalRNN` | `nn.RNN(768, 128, bidirectional)` | 1 lineal → 4 | `(B, T, 4)` | 231 K |
| `standard_rnn` (puntuación) | `PunctuationRNN` | `nn.RNN(768, 128)` | inicial → 2, final → 4 | `(B, T, 2, 4)` | 116 K |
| `bidirectional_rnn` (puntuación) | `PunctuationBidirectionalRNN` | `nn.RNN(768, 128, bidirectional)` | inicial → 2, final → 4 | `(B, T, 2, 4)` | 231 K |
| `joint_rnn` | `JointRNN` | `nn.RNN(768, 128)` | cap → 4, inicial → 2, final → 4 | `(B, T, 3, 4)` | 116 K |
| `bidirectional_joint_rnn` | `JointRNN` | `nn.RNN(768, 128, bidirectional)` | ídem | `(B, T, 3, 4)` | 232 K |
| `joint_lstm` | `JointLSTM` | `nn.LSTM(768, 128)` | ídem | `(B, T, 3, 4)` | 461 K |
| `bidirectional_joint_lstm` | `JointLSTM` | `nn.LSTM(768, 128, bidirectional)` | ídem | `(B, T, 3, 4)` | 922 K |
| `joint_projected_bigru` | `ProjectedBiGRU` | lineal 768 → 256 + GELU, `nn.GRU(256, 128, bidirectional)` | 3 MLP (256 → 128 → clases, GELU) | `(B, T, 3, 4)` | 593 K |
| `joint_residual_stacked_bilstm` | `ResidualStackedBiLSTM` | lineal 768 → 256 + GELU, `nn.LSTM(256, 128, 2 capas, bidirectional)` | 3 lineales sobre `[h_t ; token proyectado]` | `(B, T, 3, 4)` | 993 K |
| `joint_conv_bilstm` | `ConvBiLSTM` | `Conv1d(768, 256, ventana 3)` + GELU, `nn.LSTM(256, 128, bidirectional)` | 3 lineales | `(B, T, 3, 4)` | 988 K |

- Las redes de cada problema están en [`src/problem/<problema>/`](../src/problem/); las
  conjuntas en [`src/joint_rnn.py`](../src/joint_rnn.py), y sus variantes (las últimas tres
  filas, con dropout 0.3) en [`src/joint_networks.py`](../src/joint_networks.py), que explica
  qué prueba cada una.
- **Varias tareas en un tensor**: las tareas con menos clases se completan con logits `-inf`
  hasta 4 clases. Como $e^{-\infty} = 0$, en el softmax
  $P(c) = e^{z_c} / \sum_{c'} e^{z_{c'}}$ esas clases inventadas tienen probabilidad 0: nunca se predicen ni cambian la
  loss.
- **Redes conjuntas**: una sola capa recurrente compartida por las 3 tareas, con una cabeza
  lineal por tarea. Se entrenan con las 3 etiquetas (`JointDataset`, mismas instancias y splits)
  y para cada problema se evalúan solo sus cabezas. Sirven para responder si conviene compartir
  pesos entre tareas.

## Entrenamiento de las redes

Minimizan la loss de [04-metrics.md](04-metrics.md), con estos parámetros.

[`src/training/neural.py`](../src/training/neural.py):

| | |
|---|---|
| Optimizador | Adam, `LEARNING_RATE = 1e-3` |
| Épocas | `N_EPOCHS = 5`, o las de `--epochs` del pipeline |
| Batch | `BATCH_SIZE = 200` instancias |
| Recorte de gradiente | norma máxima $\tau$ = `GRAD_CLIP_NORM = 1.0`: $g \leftarrow g \cdot \min\left(1, \frac{\tau}{\lVert g \rVert_2}\right)$, con $g$ todos los gradientes juntos |
| Pesos con los que se queda | los de la época $e^* = \arg\max_e \overline{F_1}^{\,\text{test}}(e)$, con mejor F1 macro promedio en **test** (en las conjuntas, promedio de las 3 tareas) |
| Inicialización | `torch.manual_seed(12)` antes de crear cada red |
