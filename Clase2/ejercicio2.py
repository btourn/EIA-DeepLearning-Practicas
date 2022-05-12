import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    y = -(x-2)**2 + 3 +0.2*np.sin(16*x)
    return y


def loss(y, y_pred):
    diff = y - y_pred
    return np.mean(diff**2)


def predict(X, w_vec):
    return X @ w_vec


def gradient(X, y, y_pred):
    return -2*(y - y_pred) @ X



def mini_batch(X, y, w_vec, len_Xbatch, lr, epochs):
    N = len(y)
    MSE = []
    for i in range(epochs):
        for j in range(0, N, len_Xbatch):
            # Selecciono batch de filas del dataset
            Xbatch = X[j:j+len_Xbatch, :]
            # Paso forward utilizando el batch
            y_pred_j = predict(Xbatch, w_vec)
            # Paso backpropagation
            gradJ = gradient(Xbatch, y[j:j+len_Xbatch], y_pred_j)
            # Actualizacion de pesos
            w_vec = w_vec - lr*gradJ
        
        y_pred = predict(X, w_vec)        
        mse = loss(y, y_pred)
        MSE.append(mse)
        print("Epoch:", i, "MSE:", mse)

    return w_vec, MSE



n = 10000
x = np.linspace(0, 4, n)
y = fun(x)

X = np.array([x**2, x, np.ones((n,))]).T
w_vec0 = np.random.rand(3,)
len_batch = round(n/5)
lr = 0.000005
epochs = 1000
w_vec, MSE = mini_batch(X, y, w_vec0, len_batch, lr, epochs)

# Resultados
print('Parámetros originales:', w_vec0)
print('Parámetros aprendidos:', w_vec)

# Grafica MSE vs. epochs
fig1, axs1 = plt.subplots(1, sharey=True, figsize=(15, 9))
axs1.semilogy(np.arange(epochs), MSE, linewidth=1.5, color='r')
axs1.legend()
axs1.axes.set_xlabel('Epochs')
axs1.axes.set_ylabel('MSE')


# Grafica datos vs. prediccion
y_pred = predict(X, w_vec)
fig2, axs2 = plt.subplots(1, sharey=True, figsize=(15, 9))
axs2.scatter(x, y, color='b', label='Datos')
axs2.plot(x, y_pred, linewidth=3, color='r', label='Prediccion')
axs2.legend()
axs2.axes.set_xlabel('x')
axs2.axes.set_ylabel('y')

