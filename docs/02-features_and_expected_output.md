# Features y salida esperada

Qué recibe cada modelo (las features de cada instancia de [datasets.md](datasets.md)) y qué
tiene que aprender a devolver (las etiquetas). Los modelos que los usan: [models.md](models.md).

## Feature extractors

Están en [`src/features/`](../src/features/). Todos reciben la lista de tokens de una instancia.

| Extractor | `extract(tokens)` | `extract_words(tokens)` | Lo usan |
|---|---|---|---|
| `EmbeddingsExtractor` | `(n_tokens, 768)` | — | las redes |
| `AverageEmbeddingsExtractor` | `(n_tokens, 768)`, cada token con el promedio de su palabra | `(n_palabras, 768)` | `classic_embeddings` |
| `ClassicFeatureExtractor` | `(n_tokens, 48)`, cada token con los atributos de su palabra | `(n_palabras, 48)` | `classic_features*` |

- **`EmbeddingsExtractor`**: el embedding **estático** de cada token, de la tabla de
  `word_embeddings` de BERT multilingüe, como pide el enunciado (sin las capas Transformer: el
  contexto lo arman las redes). Los tokens fuera del vocabulario usan el de `[UNK]`. Con
  $E \in \mathbb{R}^{|V| \times 768}$ la tabla de embeddings y $\mathrm{id}(t)$ el índice del
  token en el vocabulario:

  $$x_t = E_{\mathrm{id}(t)} \in \mathbb{R}^{768}$$

- **`AverageEmbeddingsExtractor`**: el promedio de los embeddings de los tokens de la palabra $w$:

  $$x_w = \frac{1}{|w|} \sum_{t \in w} E_{\mathrm{id}(t)}$$

- **`ClassicFeatureExtractor`**: atributos "a la antigua", sin embeddings:
  - de la palabra (9): largo, vocales, consonantes, proporción de vocales, acentos, cantidad de
    tokens, si es interrogativa con tilde (`qué`, `cómo`, `dónde`, …), si tiene dígitos, si
    tiene signos;
  - de su posición (3): primera de la instancia, última, posición relativa;
  - de contexto (36): los 9 atributos de las 2 palabras anteriores y las 2 siguientes (ceros
    fuera de la instancia, `CONTEXT_WORDS`).

  Con $\ell$ el largo de la palabra, $v$ sus vocales, $i$ su posición y $n$ las palabras de
  la instancia, la proporción de vocales es $v / \ell$ y la posición relativa
  $i / \max(n - 1, 1)$. Si $a_i \in \mathbb{R}^9$ son los atributos propios de la palabra $i$
  y $p_i \in \mathbb{R}^3$ los de su posición, la fila de la palabra es

  $$x_i = [\, a_i \,;\, p_i \,;\, a_{i-2} \,;\, a_{i-1} \,;\, a_{i+1} \,;\, a_{i+2} \,] \in \mathbb{R}^{48}, \qquad a_j = 0 \text{ si } j \notin [0, n)$$

- **Diccionario de capitalización** ([`src/features/dictionary.py`](../src/features/dictionary.py)),
  solo en `classic_features_dictionary`: 6 atributos más por palabra, calculados con las
  palabras de **train**: si se vio, log(1 + veces vista) y la proporción de minúscula, Inicial,
  MiXta y MAYÚS. Con $n_w(c)$ las veces que la palabra $w$ apareció en train con la
  capitalización $c$ y $n_w = \sum_c n_w(c)$:

  $$d_w = \Big[\, \mathbb{1}[n_w > 0] \,,\; \log(1 + n_w) \,,\; \tfrac{n_w(0)}{n_w}, \tfrac{n_w(1)}{n_w}, \tfrac{n_w(2)}{n_w}, \tfrac{n_w(3)}{n_w} \,\Big] \in \mathbb{R}^6$$

  ($d_w = 0$ si la palabra no se vio). Para que el modelo no aprenda a confiar ciegamente en él,
  las filas de train usan un diccionario armado sin su parte de train: la oración $j$ de train
  está en la parte $k = j \bmod 5$ y sus conteos salen de $\text{train} \setminus \text{parte}_k$.
  Test, control e inferencia usan el de todo train. Se guarda dentro del `.joblib`.

## Salida esperada (etiquetas)

Cada dataset devuelve, por instancia, los tokens y una etiqueta por token
([`src/data/`](../src/data/)). Ejemplo con `"¿Cuándo vamos a McDonald's? Ellos"`:

| token | cu | ##ánd | ##o | va | ##mos | a | m | ##c | ##dona | ##ld | ' | s | ellos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capitalización | 1 | 1 | 1 | 0 | 0 | 0 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
| inicial | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| final | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

| Dataset | Forma de `y` | Regla |
|---|---|---|
| `CapitalizationDataset` | `(n_tokens,)` | la clase de la palabra original, en **todos** sus tokens |
| `PunctuationDataset` | `(n_tokens, 2)` = `[inicial, final]` | `¿` en el **primer** token de la palabra, `,` `.` `?` en el **último** |
| `JointDataset` | `(n_tokens, 3)` = `[capitalización, inicial, final]` | las dos anteriores juntas |

Detalles de las etiquetas:
- **Capitalización**: se mira solo las letras de la palabra. Una sola letra en mayúscula es
  Inicial (`"A"`); MAYÚS necesita más de una letra (`"ONU"`).
- **Puntuación**: `!` cuenta como `.` (fin de oración); `¡ ; : " ( ) -` se descartan. Si una palabra
  termina con varios signos se queda con el de mayor valor (`...?` → `?`). Un signo suelto
  (`"hola , qué"`) se pega a la palabra anterior (o a la siguiente si es de apertura).

En los batches, las etiquetas se rellenan con `IGNORE_INDEX = -100` hasta el largo de la
instancia más larga.
