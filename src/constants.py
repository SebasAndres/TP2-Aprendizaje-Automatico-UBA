
EMBEDDINGS_MODEL = 'bert-base-multilingual-cased'

DATASET_PATH = "datasets/es.txt.gz"

BATCH_SIZE = 1000

CONTROL_RATIO = 0.2

TEST_RATIO = 0.2

# Semilla de los splits: todos los modelos se comparan sobre los mismos datos
SPLIT_SEED = 42

# Signos que se remueven de los bordes de cada palabra al normalizar el texto
PUNCTUATION_MARKS = '¿?¡!.,;:"«»“”()—–-'
