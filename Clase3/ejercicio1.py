from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, x, n, w):
        X_np = np.array([x**2, x, np.ones((n,))]).T
        X = torch.from_numpy(X_np).float()
        y = -(x-2)**2 + 3 +0.2*np.sin(16*x)
        Y = torch.from_numpy(y).float()
        self.x = X
        self.y = Y
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        yhat = self.linear(x)
        return yhat.squeeze(-1)


def train(model, optimizer, criterion, trainloader, epochs):
    LOSS=[]
    #trainloss = 0.0
    for epoch in range(epochs):
        total = 0
        for x, y in trainloader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            yhat = model.forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        LOSS.append(total)
        #trainloss += loss.item()
        print("Epoch:", epoch, "MSE:", total) #trainloss/len(trainloader))
    return LOSS #trainloss 


# Definición del dataset
n = 10000
x = np.linspace(0, 4, n)
w = torch.tensor([0.52864676, 0.04550111, 0.77341328]) #torch.rand(3) 
dataset = Data(x, n, w)

# Definición de hiperparáemtros del modelo
len_batch = round(n/5)
learning_rate = 0.000005
n_epochs = 1000

# Definicion del modelo de regresion
input_size = 3
output_size = 1
model = linear_regression(input_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
trainloader = DataLoader(dataset=dataset, batch_size=len_batch)

# Entrenamiento del modelo
cost_loss = train(model, optimizer, criterion, trainloader, epochs=n_epochs)

# Grafica MSE vs. epochs
fig1, axs1 = plt.subplots(1, sharey=True, figsize=(15, 9))
axs1.semilogy(np.arange(n_epochs), cost_loss, linewidth=1.5, color='r')
axs1.legend()
axs1.axes.set_xlabel('Epochs')
axs1.axes.set_ylabel('MSE')

# Grafica datos vs. prediccion
y_pred = model.forward(dataset.x)
fig2, axs2 = plt.subplots(1, sharey=True, figsize=(15, 9))
axs2.scatter(x, dataset.y.detach().numpy(), color='b', label='Datos')
axs2.plot(x, y_pred.detach().numpy(), linewidth=3, color='r', label='Prediccion')
axs2.legend()
axs2.axes.set_xlabel('x')
axs2.axes.set_ylabel('y')