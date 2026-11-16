# Datasets

De dónde salen las instancias con las que se entrenan y evalúan todos los modelos. Qué
features y etiquetas se sacan de ellas: [features_and_expected_output.md](features_and_expected_output.md).

## De texto a instancias

Las fuentes de `datasets/` (ver [`src/data/sources.py`](../src/data/sources.py)) se cortan en
**ventanas de palabras** seguidas, sin solaparse, con un largo sorteado uniformemente entre
5 y 60 palabras (`WINDOW_*` en [`src/constants.py`](../src/constants.py), o `--window-size`):

$$L \sim \mathcal{U}\{L_{min}, \dots, L_{max}\} = \mathcal{U}\{5, \dots, 60\}$$

Con `--window-distribution normal` el largo es
$L = \mathrm{clip}\big(\mathrm{round}(\mathcal{N}(\mu, \sigma^2)), L_{min}, L_{max}\big)$, con
$\mu = \frac{L_{min} + L_{max}}{2}$ y $\sigma = \frac{L_{max} - L_{min}}{6}$.
Cada fuente aporta como máximo `CAPITALIZATION_DATASET_SIZE` / `PUNCTUATION_DATASET_SIZE`
instancias (1000 por defecto).

Cada palabra se **normaliza** como la verán los modelos (sin signos en los bordes y en
minúsculas, ver `normalize_word`) y se tokeniza con `bert-base-multilingual-cased`:

```
"¿Cuándo vamos a McDonald's?"  ->  cu ##ánd ##o va ##mos a m ##c ##dona ##ld ' s
```

Las etiquetas salen del texto original (con sus mayúsculas y signos) antes de normalizarlo.

## Palabras

Una **palabra** es un token más los `##` que lo siguen (ver `token_word_ids` y `split_words` en
[`src/words.py`](../src/words.py)). Es lo único que se puede reconstruir del CSV tokenizado de
la evaluación, así que `mcdonald's` son 3 palabras: `mcdonald`, `'` y `s`.

## Splits

Siempre los mismos (semilla `RANDOM_SEED = 12`), para que todos los modelos se comparen sobre
las mismas instancias:

```
dataset ─┬─ control (20%)              evaluación final, no se usa para elegir nada
         └─ desarrollo (80%) ─┬─ train (80%)
                              └─ test  (20%)   elige la época de cada red y el modelo ganador
```

El número para comparar modelos es el de **control**: test ya se usó para elegir.
