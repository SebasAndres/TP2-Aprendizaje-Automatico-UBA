EMBEDDINGS_MODEL = "bert-base-multilingual-cased"

# Dimensiones de las redes
EMBEDDING_DIM = 768  # tamaño de los embeddings estáticos de BERT
HIDDEN_DIM = 128  # tamaño del estado oculto h_t (hiperparámetro)

# Fuentes de texto: nombre -> ruta. Cada una tiene su lector en src/data/preprocess.py
DATASET_SOURCES = {
    "tatoeba": "datasets/es.txt.gz",
    "rioplatense": "datasets/dataset_rioplatense_completo.txt",
    "ted": "datasets/data_30TEDS.txt",
    "mrbeast": "datasets/mrbeast",
    "peliculas": "datasets/peliculas",
    "simuladores": "datasets/simuladores",
    "okupas": "datasets/okupas",
}

MODELS_DIR = "models"

# Cada modelo entrenado tiene su carpeta en models/runs/<Problema>/<score>__<modelo>/, con los pesos,
# su metadata.json (corrida <día>__<hora>, config y métricas) y sus gráficos,
# ej: models/runs/PunctuationProblem/0.812__standard_rnn/0.812__standard_rnn.pkl
RUNS_DIR = f"{MODELS_DIR}/runs"
# Lo que compara varios modelos (pósters, reportes unidos) va en models/reports/<nombre>/
REPORTS_DIR = f"{MODELS_DIR}/reports"
RUN_SEPARATOR = "__"

# Máximo de instancias que aporta CADA fuente, así la más grande (es.txt.gz) no tapa al resto
CAPITALIZATION_DATASET_SIZE = 1000
PUNCTUATION_DATASET_SIZE = 1000

# Peso de cada clase en la CrossEntropy de las redes: (N / (K · n_clase)) ** CLASS_WEIGHT_POWER.
# 1 = cada clase pesa lo mismo, 0.5 = raíz (punto medio), 0 = sin pesos. Ver src/training/weights.py
CLASS_WEIGHT_POWER = 0.25

# Las instancias son ventanas de palabras sin solapamiento, con un largo sorteado entre
# WINDOW_MIN_SIZE y WINDOW_MAX_SIZE según WINDOW_DISTRIBUTION ("uniform" o "normal")
WINDOW_MIN_SIZE = 5
WINDOW_MAX_SIZE = 60
WINDOW_DISTRIBUTION = "uniform"

BATCH_SIZE = 200

CONTROL_RATIO = 0.2

TEST_RATIO = 0.2

# Todos los modelos se comparan sobre los mismos datos y se inicializan con la misma SEED
RANDOM_SEED = 12

# Signos que se remueven de los bordes de cada palabra al normalizar el texto
PUNCTUATION_MARKS = '¿?¡!.,;:"«»“”()—–-'
