import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import ldl

def lector_csv(nombre_archivo: str, nombre_variables: list[str]):
    """
    helper para leer csv y seleccionar variables.
    """
    all_x = pd.read_csv(nombre_archivo)
    selected_x = all_x[nombre_variables]
    return selected_x

def A_matrix(length_features: int): 
    """
    Genera la matriz de trancisión A a partir del # de featurs.
    """
    length_features *= 2
    A = np.zeros((length_features, length_features))
    for i in range(0,length_features,2):
        A[i,i+1] = 1
    return A

def taylor_series(grad: int, A_matrix: np.ndarray, freq: int):
    """
    regresa la serie de taylor a partir de la matriz A
    """
    time = 1/freq
    A_t = A_matrix*time
    I = np.eye(A_t.shape[1])
    F = I + (A_t)
    A2 =  A_t.copy()
    for i in range(1,grad,1):
        A2 = A2 @ A_t
        if not np.any(A2):
            break
        F = F + A2/(i + 1)
    return F 

def H_matrix(length_features: int):
    """
    Calcula la matriz de observación H.
    """
    H = np.zeros((length_features, length_features*2))
    for i in range(length_features):
        H[i,2*i] = 1
    return H

def Q_matrix(length_features: int):
    """
    Covarianza del ruido del proceso.
    """
    length_features *= 2
    Q = np.eye(length_features)
    for i in range(length_features):
        if (i+1)%2 == 0:
            Q[i,i] = 0.01
        else:
            Q[i,i] = 0.0001
    return Q

def R_matrix(length_features: int):
    """
    Covarianza del ruido de la observación.
    """
    R = np.eye(length_features) * 0.0
    return R

def x_state(length_features: int,data):
    """
    el estad inicial 
    """
    length_features *= 2
    x_state = np.zeros((length_features,1))
    data = data.to_numpy()
    for i in range(0,length_features,2):
        x_state[i] = data[0][int(i/2)]
    return x_state

def P_matrix(length_features: int):
    """
    Calcula la matriz de covarianza P.
    """
    P = np.eye(length_features*2)*0.5
    return P

def S_matrix(P: np.ndarray):
    """
    Calcula la matriz S a partir de LDL transpuesta.
    """
    L, D, perm = ldl(P, lower=True)
    S = L @ np.sqrt(D)
    return S

def U_matrix(F: np.ndarray, S: np.ndarray, Q: np.ndarray):
    """
    Construye la matriz U a partir de las matrices F, S y Q.
    """
    Q_sqrt = np.diag(np.sqrt(np.diag(Q)))
    U = np.vstack((F@S,Q_sqrt))
    return U

def Givens_rotation(U: np.ndarray):
    """
    Aplica rotaciones de Givens para triangularizar la matriz U."""
    n_cols = U.shape[1]
    n_rows = U.shape[0]

    for j in range(n_cols):
        for i in range(n_rows-2, j-1, -1):  
            a = U[i, j]
            b = U[i+1, j]

            if b == 0:
                c, s = 1, 0
            else:
                r = np.hypot(a, b)   
                c, s = a/r, b/r

 
            Gt = np.array([[c, s], [-s, c]])
            U[i:i+2, :] = Gt @ U[i:i+2, :]

    S = np.triu(U[:n_cols, :n_cols])  
    return S

def chol_downdate_upper_safe(R: np.ndarray, u: np.ndarray, eps=1e-12):
    """
    Actualiza la matriz de covarianza R utilizando el vector u.
    """
    R = R.copy()
    u = u.reshape(-1).astype(float)
    n = R.shape[0]
    denom = np.abs(u) + eps
    ratios = np.where(denom > 0, R.diagonal() / denom, np.inf)
    gamma_max = float(np.min(ratios))
    gamma = min(1.0, 0.999 * gamma_max)
    u *= gamma
    for k in range(n):
        rkk = R[k, k]; uk = u[k]
        r2 = rkk*rkk - uk*uk
        if r2 <= eps: r2 = eps
        r = np.sqrt(r2)
        c = r / rkk; s = uk / rkk
        R[k, k] = r
        if k+1 < n:
            Rkj = R[k, k+1:].copy()
            u_tail = u[k+1:].copy()
            R[k, k+1:] = (Rkj - s*u_tail) / c
            u[k+1:] = c*u_tail - s*Rkj
    return R

def potters_algorithm(X_pred: np.ndarray, S_pred: np.ndarray, y_vec: np.ndarray, H: np.ndarray, R: np.ndarray):
    """
    Algoritmo de Potter para la actualización del estado y la covarianza en el filtro de Kalman.
    """
    X, S = X_pred, S_pred
    m = y_vec.shape[0]
    for i in range(m):
        h_i = H[i, :].reshape(-1, 1)         
        r_ii = float(R[i, i])              
        y_i = float(y_vec[i, 0])            
        w = S.T @ h_i                       
        Sw = S @ w                          
        alpha = float((w.T @ w).item()) + r_ii
        if alpha <= 1e-12: alpha = 1e-12
        K = Sw / alpha                   
        res = y_i - float((h_i.T @ X).item())
        X = X + K * res
        u = Sw / np.sqrt(alpha)             
        S = chol_downdate_upper_safe(S, u)   
    return X, S


def kalman(data: pd.DataFrame, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, XO: np.ndarray, SO: np.ndarray):
    """
    Algoritmo de filtro de kalman en ensamble con suavizado usando el algoritmo de Potter.
    """
    X_est = []      
    X = XO.copy()
    S = SO.copy()
    
    for t in range(len(data)):
         w = np.random.multivariate_normal(np.zeros(F.shape[0]), Q).reshape(-1, 1)
         X_pred = F @ X + w
         U = U_matrix(F,S,Q)
         S_pred = Givens_rotation(U)
         Y_t = data.iloc[t].values.reshape(-1, 1)
         v = np.random.multivariate_normal(np.zeros(H.shape[0]), R).reshape(-1, 1)
         Y_t_noisy = Y_t + v
         X, S = potters_algorithm(X_pred, S_pred, Y_t, H, R)
         X_est.append(X.flatten())
    return np.array(X_est)
    

        
    

data = lector_csv("User1_Pre2.csv",["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6", "F4", "F8", "AF4"])
length_features = data.shape[1]
A = A_matrix(length_features) # Matriz A para el modelo de estado
F = taylor_series(2,A,128)  # Matriz de transición de estado según la serie de Taylor
H = H_matrix(length_features) # Matriz de observación
Q = Q_matrix(length_features) # Covarianza del ruido del proceso
R = R_matrix(length_features) # Covarianza del ruido de la observación
X = x_state(length_features,data) # Estado inicial
P = P_matrix(length_features) # Covarianza inicial del estado
S = S_matrix(P) # Matriz de S calculada con LDL transpuesta

X_est = kalman(data,F,H,Q,R,X,S)


# graficar cada señal real vs estimada por el filtro
plt.figure(figsize=(12,6))
for i, col in enumerate(data.columns):
    plt.plot(data.iloc[:, i], label=f"Real {col}")
    plt.plot(X_est[:, 2*i], '--', label=f"Kalman {col}")  
plt.legend()
plt.title("Filtro de Kalman - Señales reales vs estimadas")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Amplitud")


plt.ylim(-100, 100)

plt.show()
