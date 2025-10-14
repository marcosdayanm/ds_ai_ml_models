import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import ldl

def lector_csv(nombre_archivo, nombre_variables):
    all_x = pd.read_csv(nombre_archivo)
    selected_x = all_x[nombre_variables]
    return selected_x

def A_matrix(lenght_features):
    lenght_features *= 2
    A = np.zeros((lenght_features, lenght_features))
    for i in range(0,lenght_features,2):
        A[i,i+1] = 1
    return A

def taylor_series(grad, A_matrix,freq):
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

def H_matrix(lenght_features):
    H = np.zeros((lenght_features, lenght_features*2))
    for i in range(lenght_features):
        H[i,2*i] = 1
    return H

def Q_matrix(lenght_features):
    lenght_features *= 2
    Q = np.eye(lenght_features)
    for i in range(lenght_features):
        if (i+1)%2 == 0:
            Q[i,i] = 0.0
        else:
            Q[i,i] = 0.000
    return Q

def R_matrix(lenght_features):
    R = np.eye(lenght_features) * 0.019
    return R

def x_state(lenght_features,data):
    lenght_features *= 2
    x_state = np.zeros((lenght_features,1))
    data = data.to_numpy()
    for i in range(0,lenght_features,2):
        x_state[i] = data[0][int(i/2)]
    return x_state

def P_matrix(lenght_features):
    P = np.eye(lenght_features*2)*0.5
    return P

def S_matrix(P):
    L, D, perm = ldl(P, lower=True)
    S = L @ np.sqrt(D)
    return S

def U_matrix(F,S,Q):
    Q_sqrt = np.diag(np.sqrt(np.diag(Q)))
    U = np.vstack((F@S,Q_sqrt))
    return U

def Givens_rotation(U):
    n_cols = U.shape[1]
    n_rows = U.shape[0]

    for j in range(n_cols):
        for i in range(n_rows-2, j-1, -1):  # de abajo hacia arriba
            a = U[i, j]
            b = U[i+1, j]

            if b == 0:
                c, s = 1, 0
            else:
                r = np.hypot(a, b)   # sqrt(a^2 + b^2)
                c, s = a/r, b/r

            # aplica rotación correcta (G^T U)
            Gt = np.array([[c, s], [-s, c]])
            U[i:i+2, :] = Gt @ U[i:i+2, :]

    S = np.triu(U[:n_cols, :n_cols])  # triangular superior
    return S

def chol_downdate_upper_safe(R, u, eps=1e-12):
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

def potters_algorithm(X_pred, S_pred, y_vec, H, R):
    X, S = X_pred, S_pred
    m = y_vec.shape[0]
    for i in range(m):
        h_i = H[i, :].reshape(-1, 1)         # (n,1)
        r_ii = float(R[i, i])                # escalar
        y_i = float(y_vec[i, 0])             # escalar
        w = S.T @ h_i                        # (n,1)
        Sw = S @ w                           # (n,1)
        alpha = float((w.T @ w).item()) + r_ii
        if alpha <= 1e-12: alpha = 1e-12
        K = Sw / alpha                       # (n,1)
        res = y_i - float((h_i.T @ X).item())
        X = X + K * res
        u = Sw / np.sqrt(alpha)              # (n,1)
        S = chol_downdate_upper_safe(S, u)   # mantiene triangular y SPD
    return X, S


def kalman(data,F,H,Q,R,XO,SO):
    X_est = []          # lista para guardar estados estimados
    X = XO.copy()
    S = SO.copy()
    for t in range(len(data)):
         X_pred = F @ X
         U = U_matrix(F,S,Q)
         S_pred = Givens_rotation(U)
         Y_t = data.iloc[t].values.reshape(-1, 1)
         X, S = potters_algorithm(X_pred, S_pred, Y_t, H, R)
         X_est.append(X.flatten())
    return np.array(X_est)
    

        
    

data = lector_csv("User1_Pre2.csv",["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6"])
lenght_features = data.shape[1]
A = A_matrix(lenght_features)
F = taylor_series(2,A,128) #
H = H_matrix(lenght_features) #
Q = Q_matrix(lenght_features) #
R = R_matrix(lenght_features) #
X = x_state(lenght_features,data) #
P = P_matrix(lenght_features)
S = S_matrix(P) #

X_est = kalman(data,F,H,Q,R,X,S)
#print("---- ESTIMACIÓN COMPLETA ----")
#print(X_est)


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
