
EMBEDDINGS_MODEL = 'bert-base-multilingual-cased'

DATASET_PATH = "datasets/es.txt.gz"

MODELS_DIR = "models"

# Los modelos se guardan en models/<Problema>/<día>/<score>__<hora>__<modelo>,
# ej: models/PunctuationProblem/2026-11-12/0.812__19-30__standard_rnn.pkl
RUN_SEPARATOR = "__"

CAPITALIZATION_DATASET_SIZE=5000
PUNCTUATION_DATASET_SIZE=5000

BATCH_SIZE = 200

CONTROL_RATIO = 0.2

TEST_RATIO = 0.2

# Todos los modelos se comparan sobre los mismos datos y se inicializan con la misma SEED
RANDOM_SEED = 12

# Signos que se remueven de los bordes de cada palabra al normalizar el texto
PUNCTUATION_MARKS = '¿?¡!.,;:"«»“”()—–-'
