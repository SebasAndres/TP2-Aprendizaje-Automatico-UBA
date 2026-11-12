
EMBEDDINGS_MODEL = 'bert-base-multilingual-cased'

DATASET_PATH = "datasets/es.txt.gz"

MODELS_DIR = "models"

# Cada ejecución del pipeline guarda sus modelos en models/<día>__<hora>/<Problema>/<score>__<modelo>,
# ej: models/2026-11-12__19-30-05/PunctuationProblem/0.812__standard_rnn.pkl
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
