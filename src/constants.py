
EMBEDDINGS_MODEL = 'bert-base-multilingual-cased'

DATASET_PATH = "datasets/es.txt.gz"

CAPITALIZATION_DATASET_SIZE=30000
PUNCTUATION_DATASET_SIZE=30000

BATCH_SIZE = 1000

CONTROL_RATIO = 0.2

TEST_RATIO = 0.2

# Todos los modelos se comparan sobre los mismos datos y se inicializan con la misma SEED
RANDOM_SEED = 12

# Signos que se remueven de los bordes de cada palabra al normalizar el texto
PUNCTUATION_MARKS = '¿?¡!.,;:"«»“”()—–-'
