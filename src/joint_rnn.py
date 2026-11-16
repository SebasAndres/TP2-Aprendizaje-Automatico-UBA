"""Una sola RNN que predice capitalización y puntuación al mismo tiempo.

La RNN se comparte entre las tres tareas y cada tarea tiene su propia "cabeza" lineal:
- capitalización: 4 clases (0, 1, 2, 3)
- puntuación inicial: 2 clases ("", ¿)
- puntuación final: 4 clases ("", ",", ".", "?")

`JointLSTM` es la misma red con celdas LSTM en lugar de la RNN simple: cada celda tiene
compuertas que deciden qué recordar y qué olvidar, así que conserva mejor el contexto lejano
(ej: el "¿" del principio de una pregunta larga para poner el "?" al final).

En el pipeline son los experimentos `joint_rnn`, `bidirectional_joint_rnn`, `joint_lstm` y
`bidirectional_joint_lstm`: se entrenan con las 3 tareas sobre las mismas instancias de train
que el resto de los modelos, y para cada problema se evalúan solo sus cabezas. También se
pueden correr solas, con las 3 tareas a la vez:

    uv run python -m src.joint_rnn                      # todas las instancias de cada fuente
    uv run python -m src.joint_rnn --size 30000         # como mucho 30000 instancias por fuente
    uv run python -m src.joint_rnn --bidirectional      # RNN bidireccional
    uv run python -m src.joint_rnn --lstm               # LSTM en lugar de RNN simple
"""
import argparse
import logging

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.constants import CLASS_WEIGHT_POWER, EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED
from src.data import (
    CapitalizationDataset,
    PunctuationDataset,
    build_loader,
    load_documents,
    load_samples,
    split_dataset_dev_control,
    split_train_test,
)
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.data.samples import add_window_args, window_from_args
from src.features import EmbeddingsExtractor
from src.metrics import JointMetric
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES

logger = logging.getLogger("src.joint_rnn")


class JointDataset(CapitalizationDataset, PunctuationDataset):
    """Cada token tiene 3 etiquetas: <capitalización, puntuación inicial, puntuación final>"""

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        capitalization = CapitalizationDataset._expected_output(self, words, word_tokens)  # (n_tokens,)
        punctuation = PunctuationDataset._expected_output(self, words, word_tokens)        # (n_tokens, 2)
        return torch.cat([capitalization.unsqueeze(1), punctuation], dim=1)                # (n_tokens, 3)


class JointRNN(nn.Module):
    """RNN compartida + una cabeza lineal por tarea: capitalización y puntuación a la vez.

    La RNN calcula el estado oculto ``h_t`` de cada token y tres cabezas lineales lo leen:

    - capitalización: 4 clases (minúscula, Inicial, MiXta, MAYÚS)
    - puntuación inicial: 2 clases (``""``, ``"¿"``)
    - puntuación final: 4 clases (``""``, ``","``, ``"."``, ``"?"``)

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token (768 para BERT multilingüe).
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección.
    bidirectional : bool, default=False
        Si es ``True`` lee la oración en las dos direcciones y las cabezas reciben
        ``[h→_t ; h←_t]``, de dimensión ``2 * hidden_dim``.

    Attributes
    ----------
    rnn : torch.nn.RNN or torch.nn.LSTM
        Capa recurrente compartida, creada por ``_recurrent_layer``.
    capitalization_head : torch.nn.Linear
        Cabeza de capitalización, a ``N_CAPITALIZATION_CLASSES`` clases.
    initial_head : torch.nn.Linear
        Cabeza de puntuación inicial, a ``len(INITIAL_LABELS)`` clases.
    final_head : torch.nn.Linear
        Cabeza de puntuación final, a ``len(FINAL_LABELS)`` clases.

    See Also
    --------
    JointLSTM : la misma red con celdas LSTM.
    JointDataset : dataset con las 3 etiquetas por token que espera esta red.

    Notes
    -----
    Las tareas con menos clases se completan con ``-inf`` hasta ``max_clases`` (ver
    ``_pad_classes``) para devolver un único tensor. Devuelve logits, sin softmax.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        bidirectional: bool = False,
    ):
        super().__init__()
        # Parte compartida: lee la oración y construye el estado oculto h_t de cada token
        self.rnn = self._recurrent_layer(input_dim, hidden_dim, bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # Una cabeza por tarea, todas leen el mismo h_t
        self.capitalization_head = nn.Linear(out_dim, N_CAPITALIZATION_CLASSES)
        self.initial_head = nn.Linear(out_dim, len(INITIAL_LABELS))
        self.final_head = nn.Linear(out_dim, len(FINAL_LABELS))

    @staticmethod
    def _recurrent_layer(input_dim: int, hidden_dim: int, bidirectional: bool) -> nn.Module:
        return nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Calcula los logits de las 3 tareas para cada token.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_tokens, input_dim)
            Embeddings de las oraciones, con padding al final hasta ``n_tokens``.
        lengths : list[int]
            Cantidad de tokens reales de cada oración (``len(lengths) == batch``).

        Returns
        -------
        torch.Tensor of shape (batch, n_tokens, 3, max_clases)
            Logits por token, tarea y clase. Las tareas van en el orden
            capitalización, puntuación inicial, puntuación final, y las clases que
            no existen en una tarea valen ``-inf``.
        """
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)  # la LSTM devuelve (h_n, c_n) en lugar de h_n: se descarta igual
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        heads = [self.capitalization_head(h), self.initial_head(h), self.final_head(h)]
        return torch.stack(_pad_classes(heads), dim=2)


class JointLSTM(JointRNN):
    """La misma red conjunta de ``JointRNN``, con celdas LSTM.

    Además del estado oculto ``h_t`` lleva una memoria ``c_t`` que solo cambia a
    través de compuertas (olvidar, escribir, leer), así que conserva mejor el
    contexto lejano. Con el mismo ``hidden_dim`` tiene ~4 veces los parámetros
    recurrentes de la RNN simple.

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token.
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto (y de la memoria) de cada dirección.
    bidirectional : bool, default=False
        Si es ``True`` lee la oración en las dos direcciones.

    See Also
    --------
    JointRNN : misma interfaz y misma salida.
    """

    @staticmethod
    def _recurrent_layer(input_dim: int, hidden_dim: int, bidirectional: bool) -> nn.Module:
        return nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)


def _pad_classes(logits: list[torch.Tensor]) -> list[torch.Tensor]:
    """Completa con ``-inf`` las tareas con menos clases para poder apilarlas.

    Con ``-inf`` la clase inventada tiene probabilidad 0: nunca se predice ni cambia la loss.

    Parameters
    ----------
    logits : list[torch.Tensor]
        Logits de cada tarea, de shape ``(..., n_classes_tarea)``. Todas las
        dimensiones salvo la última deben coincidir.

    Returns
    -------
    list[torch.Tensor]
        Los mismos logits, todos de shape ``(..., max_clases)``.

    Examples
    --------
    >>> padded = _pad_classes([torch.zeros(1, 2), torch.zeros(1, 4)])
    >>> [task.shape for task in padded]
    [torch.Size([1, 4]), torch.Size([1, 4])]
    >>> padded[0]
    tensor([[0., 0., -inf, -inf]])
    """
    n_classes = max(task.size(-1) for task in logits)
    return [
        nn.functional.pad(task, (0, n_classes - task.size(-1)), value=float("-inf"))
        for task in logits
    ]


def create_joint_rnn_model(bidirectional: bool = False) -> JointRNN:
    """Crea la RNN conjunta, sin entrenar, con la semilla fija.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la RNN lee la oración en las dos direcciones.

    Returns
    -------
    JointRNN
        Red con las dimensiones por defecto de ``src.constants``.
    """
    torch.manual_seed(RANDOM_SEED)  # misma inicialización de pesos en cada corrida
    return JointRNN(bidirectional=bidirectional)


def create_joint_lstm_model(bidirectional: bool = False) -> JointLSTM:
    """Crea la LSTM conjunta, sin entrenar, con la semilla fija.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la LSTM lee la oración en las dos direcciones.

    Returns
    -------
    JointLSTM
        Red con las dimensiones por defecto de ``src.constants``.
    """
    torch.manual_seed(RANDOM_SEED)
    return JointLSTM(bidirectional=bidirectional)


def create_joint_model(bidirectional: bool = False, lstm: bool = False) -> JointRNN:
    """Crea la red conjunta pedida, sin entrenar.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la red lee la oración en las dos direcciones.
    lstm : bool, default=False
        Si usa celdas LSTM (``JointLSTM``) en lugar de la RNN simple (``JointRNN``).

    Returns
    -------
    JointRNN
        Una ``JointRNN`` o una ``JointLSTM`` (que hereda de ``JointRNN``).
    """
    return create_joint_lstm_model(bidirectional) if lstm else create_joint_rnn_model(bidirectional)


def joint_model_name(bidirectional: bool = False, lstm: bool = False) -> str:
    """Nombre del experimento, con el que se guarda el archivo. Ej: 'bidirectional_joint_lstm'"""
    return ("bidirectional_" if bidirectional else "") + ("joint_lstm" if lstm else "joint_rnn")


def _loaders(dataset: Dataset) -> tuple[DataLoader, DataLoader]:
    """Mismos splits y semillas que el pipeline"""
    development_set, _ = split_dataset_dev_control(dataset)
    train_set, test_set = split_train_test(development_set)
    logger.info("Instancias: train=%d, test=%d", len(train_set), len(test_set))
    return build_loader(train_set, shuffle=True), build_loader(test_set, shuffle=False)


def main() -> None:
    # Import local: src.training importa este módulo para el experimento joint_rnn
    from src.training.neural import N_EPOCHS, train_neural_model
    from src.training.result import format_score

    parser = argparse.ArgumentParser(description="Entrena una RNN conjunta de capitalización y puntuación")
    parser.add_argument("--size", type=int, default=None, help="usar como mucho N instancias de cada fuente (default: todas)")
    add_window_args(parser)
    parser.add_argument("--class-weight-power", type=float, default=CLASS_WEIGHT_POWER,
                        help=f"peso de las clases raras en la loss: 1 = balanceado, 0 = sin pesos (default: {CLASS_WEIGHT_POWER})")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help=f"épocas de entrenamiento (default: {N_EPOCHS})")
    parser.add_argument("--bidirectional", action="store_true", help="usar una RNN bidireccional")
    parser.add_argument("--lstm", action="store_true", help="usar celdas LSTM en lugar de la RNN simple")
    parser.add_argument("--no-save", action="store_true", help="no guardar la red entrenada en models/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    samples = load_samples(load_documents(), args.size, window_from_args(args))
    dataset = JointDataset(samples)
    train_loader, test_loader = _loaders(dataset)
    model = create_joint_model(args.bidirectional, args.lstm)
    name = joint_model_name(args.bidirectional, args.lstm)
    result = train_neural_model(
        train_loader, test_loader, EmbeddingsExtractor(), JointMetric(), model,
        n_epochs=args.epochs, class_weight_power=args.class_weight_power
    )
    print(f"\n{name} - F1 macro sobre test: {format_score(result.score)}")
    if not args.no_save:
        save_joint_result(result, name)


def save_joint_result(result, name: str) -> None:
    """Guarda la red como lo hace el pipeline, en models/<día>__<hora>/<Problema>/<score>__<name>.pkl,
    una copia por problema con el score de sus tareas. Así src.inference la encuentra al elegir
    cualquiera de los dos problemas, y como es conjunta predice los dos"""
    from src.problem import CapitalizationProblem, PunctuationProblem
    from src.training.train import _file_stem, _save_experiment, create_run_directory, problem_dir

    run_dir = create_run_directory()
    for problem in (CapitalizationProblem(), PunctuationProblem()):
        problem_result = result.only_tasks(problem.metric.tasks)
        model_dir = problem_dir(run_dir, problem)
        file_stem = _file_stem(model_dir, name, problem_result)
        _save_experiment(name, problem_result, file_stem, type(problem).__name__)
        print(f"Guardado en {file_stem}.pkl")


if __name__ == "__main__":
    main()
