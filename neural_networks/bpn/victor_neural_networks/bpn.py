import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt


def sigmoidal(z: np.ndarray) -> np.ndarray:
    """Sigmoide estable numéricamente."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidalGradiente(z: np.ndarray) -> np.ndarray:
    """Gradiente de la sigmoide g'(z) = g(z) * (1 - g(z))."""
    s = sigmoidal(z)
    return s * (1.0 - s)

def randInicializacionPesos(L_in: int, L_out: int, epsilon: float = 0.12) -> np.ndarray:
    """
    Inicializa W con valores desde [-epsilon, epsilon].
    - L_in: número de entradas (sin bias) a esta capa.
    - L_out: número de unidades (neuronas) en esta capa.
    regresa matriz de forma (L_out, L_in).
    """
    rng = np.random.default_rng()
    return rng.uniform(low=-epsilon, high=+epsilon, size=(L_out, L_in))


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga archivo de dataset, acepta cualquier dimensión de entrada, la salida es un número
    regresa:
        X: (m, n) float
        y: (m,) int
    """
    data = np.loadtxt(path)
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(int)
    return X, y

def one_hot_y(y: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Convierte etiquetas {1..10} a codificación one-hot de 10 columnas.
    La columna k-1 es 1 para etiqueta k (etiqueta 10 corresponde a la columna 9).

    Ésto para poder representar cada número de los outputs con un vector de 10 dimensiones y poder comparar con la neurona que se prenda en la capa de salida
    """
    m = y.shape[0]
    Y = np.zeros((m, num_labels), dtype=np.float64)
    Y[np.arange(m), y - 1] = 1.0
    return Y


def forward(X: np.ndarray,
            W1: np.ndarray, b1: np.ndarray,
            W2: np.ndarray, b2: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Propaga, hace las operaciones en cada capa como si se evaluara un polinomio
    Luego aplica función de activación a cada capa.
    Regresa los pesos por cada capa

    Usé ChatGPT5 para operar con fórmulas de matrices
    """
    Z1 = X @ W1.T + b1 
    A1 = sigmoidal(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = sigmoidal(Z2)
    nn_state = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return nn_state


def costo_logistico(A2: np.ndarray, Y: np.ndarray, W1: np.ndarray, W2: np.ndarray, lambda_: float = 0.0) -> float:
    """
    Costo promedio por ejemplo para clasificación multiclase con salidas sigmoides independientes.
    J = -(1/m) sum( Y*log(A2) + (1-Y)*log(1-A2) ) + regularización L2 (opcional, sin biases)
    """
    m = Y.shape[0]
    eps = 1e-12
    # Término de entropía cruzada
    CE = -(Y * np.log(A2 + eps) + (1.0 - Y) * np.log(1.0 - A2 + eps)).sum() / m
    # Regularización L2 (solo pesos, no biases)
    if lambda_ > 0.0:
        reg = (lambda_ / (2.0 * m)) * (np.sum(W1 * W1) + np.sum(W2 * W2))
        return CE + reg
    return CE


def backprop(nn_state: Dict[str, np.ndarray],
             Y: np.ndarray,
             W1: np.ndarray, b1: np.ndarray,
             W2: np.ndarray, b2: np.ndarray,
             lambda_: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula gradientes de J respecto a W1, b1, W2, b2 usando backpropagation vectorizado.
    Devuelve dW1, db1, dW2, db2.

    Usé ChatGPT para transformar de fórmulas que operaban elemento por elemento de las matrices, a hacer operaciones con matrices
    """
    X, Z1, A1, Z2, A2 = nn_state["X"], nn_state["Z1"], nn_state["A1"], nn_state["Z2"], nn_state["A2"]
    m = X.shape[0]

    # Capa de salida
    delta2 = A2 - Y

    # Gradientes segunda capa
    dW2 = (delta2.T @ A1) / m
    db2 = delta2.mean(axis=0)
    if lambda_ > 0.0:
        dW2 += (lambda_ / m) * W2

    # Propagación a capa oculta
    delta1 = (delta2 @ W2) * sigmoidalGradiente(Z1) 

    # Gradientes capa oculta
    dW1 = (delta1.T @ X) / m
    db1 = delta1.mean(axis=0)
    if lambda_ > 0.0:
        dW1 += (lambda_ / m) * W1

    return dW1, db1, dW2, db2


def entrenaRN(input_layer_size: int,
              hidden_layer_size: int,
              num_labels: int,
              X: np.ndarray,
              y: np.ndarray,
              alpha: float = 1.0,
              epocas: int = 400,
              lambda_: float = 0.0,
              verbose_cada: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """
    Entrena una RN 3 capas con gradient descent full-batch.
    Regresa:
      W1, b1, W2, b2, history (costo y accuracy por época para graficar)
    """
    # Inicialización de pesos
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size, epsilon=0.12)
    b1 = np.zeros(hidden_layer_size, dtype=np.float64)                         
    W2 = randInicializacionPesos(hidden_layer_size, num_labels, epsilon=0.12)    
    b2 = np.zeros(num_labels, dtype=np.float64)                             

    # One-hot para y
    Y = one_hot_y(y, num_labels)  # (m,10)

    history = {"J": [], "acc": []}

    for e in range(1, epocas + 1):
        # Forward
        nn_state = forward(X, W1, b1, W2, b2)
        # Backpropagation
        dW1, db1, dW2, db2 = backprop(nn_state, Y, W1, b1, W2, b2, lambda_)
        
        # Costo y accuracy
        J = costo_logistico(nn_state["A2"], Y, W1, W2, lambda_)
        y_pred = np.argmax(nn_state["A2"], axis=1) + 1
        acc = (y_pred == y).mean()
        #Guardar históricos
        history["J"].append(J)
        history["acc"].append(acc)

        # Gradient descent a matrices de pesos y bias, por capas
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2

        if verbose_cada and (e % verbose_cada == 0 or e == 1 or e == epocas):
            print(f"[época {e:4d}] J={J:.6f} | acc={acc*100:.2f}%")

    return W1, b1, W2, b2, history


def prediceRNYaEntrenada(X: np.ndarray,
                         W1: np.ndarray, b1: np.ndarray,
                         W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Realiza forward y devuelve etiquetas {1..10} (10 representa el dígito '0').
    """
    A2 = forward(X, W1, b1, W2, b2)["A2"]
    y_pred = np.argmax(A2, axis=1) + 1
    return y_pred


def plot_acc_and_error(hist):
    """
    Grafica accuracy ('acc') y error/costo ('j') por época en la misma ventana.

    Usé GitHub Copilot para crear las gráficas
    """
    epochs = range(1, len(hist['acc']) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    axs[0].plot(epochs, hist['acc'], marker='o', color='green')
    axs[0].set_title('Accuracy por época')
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)
    
    axs[1].plot(epochs, hist['J'], marker='o', color='red')
    axs[1].set_title('Error total por época')
    axs[1].set_xlabel('Época')
    axs[1].set_ylabel('Error total')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath = "digitos.txt"

    X, y = load_dataset(filepath)
    print("Datos:", X.shape, y.shape, "| valores y:", np.unique(y))

    W1, b1, W2, b2, hist = entrenaRN(
        input_layer_size=400,
        hidden_layer_size=25,
        num_labels=10,
        X=X,
        y=y,
        alpha=0.6, 
        epocas=1000, 
    )

    y_hat = prediceRNYaEntrenada(X, W1, b1, W2, b2)
    acc = (y_hat == y).mean()
    print(f"Accuracy sobre entrenamiento: {acc*100:.2f}%")

    plot_acc_and_error(hist)
