import torch 
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)


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


class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer 
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        # Define the first linear layer as an attribute, this is not good practice
        self.a1 = None
        self.l1 = None
        self.l2 = None
    # Prediction
    def forward(self, x):
        self.l1 = self.linear1(x)
        self.a1 = sigmoid(self.l1)
        self.l2=self.linear2(self.a1)
        yhat = self.l2 #sigmoid(self.linear2(self.a1))
        return yhat.squeeze(-1)


def PlotStuff(X, dataset, model, epoch, leg=True):
    
    plt.plot(X, model(dataset.x).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X, dataset.y.numpy(), 'r')
    plt.xlabel('x')
    if leg == True:
        plt.legend()
    else:
        pass


def train(X, dataset, model, optimizer, criterion, trainloader, epochs=1000):
    cost = []
    total=0
    for epoch in range(epochs):
        total=0
        for x, y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #cumulative loss 
            total+=loss.item() 
        cost.append(total)
        print("Epoch:", epoch, "MSE:", total)
        if epoch % 300 == 0:    
            PlotStuff(X, dataset, model, epoch, leg=True)
            plt.show()
            model(dataset.x)
            plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=dataset.y.numpy().reshape(-1))
            plt.title('activations')
            plt.show()
    return cost


# Definición del dataset
n = 10000
x = np.linspace(0, 4, n)
w = torch.tensor([0.52864676, 0.04550111, 0.77341328]) #torch.rand(3) 
dataset = Data(x, n, w)

# Definición de hiperparáemtros del modelo
len_batch = round(n/5)
learning_rate = 0.000005
n_epochs = 1000


# Train the model
# size of input 
D_in = 3
# size of hidden layer 
H = 2
# number of outputs 
D_out = 1
# learning rate 
learning_rate = 0.1
# create the model 
model = Net(D_in, H, D_out)
#optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
trainloader = DataLoader(dataset=dataset, batch_size=len_batch)

#train the model usein
cost_MSE = train(x, dataset, model, optimizer, criterion, trainloader, epochs=n_epochs)

#plot the loss
plt.plot(cost_MSE)
plt.xlabel('epoch')
plt.title('cross entropy loss')