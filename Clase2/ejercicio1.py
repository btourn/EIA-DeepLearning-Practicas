import numpy as np

def fun(x, grad=False):
    y = -(x-2)**2 + 3
    dy_dx = []
    if grad:
        dy_dx = -2*(x-2)

    return y, dy_dx

lr = 0.5
epochs = 500
tol = 1e-6
x0 = float(np.random.uniform(low=-10, high=10))
x = np.copy(x0)
for i in range(epochs):
    
    f, gradf = fun(x, grad=1)
    if abs(gradf)<tol:
        print("Minimum reached. x:", x, "Epoch:", i, '; x:', x, '; f:', f, '; gradf:', gradf)
        break

    x = x + lr*gradf

    print("Epoch:", i, '; x:', x, '; f:', f, '; gradf:', gradf)

