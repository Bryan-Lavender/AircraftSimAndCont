import numpy as np
import matplotlib.pyplot as plt
import scipy

A = [[0,1],[-0.01,-0.1]]
X0 = [-10, 20]
print("Eigenvalues: ",np.linalg.eigvals(A))
print("Eigenvectors: ",np.linalg.eig(A)[1])
eigenvalues, eigenvectors = np.linalg.eig(A)
def model_func(t):
    exMatrix = scipy.linalg.expm(np.array(A)*t)
    return exMatrix @ np.array(X0)

alpha = eigenvalues.real[0]
beta = eigenvalues.imag[0]

wn = np.sqrt(alpha**2 + beta**2)
dr = -alpha/wn

print(wn, dr)




time_points = np.linspace(0, 120, 500)
trajectory = np.array([model_func(t) for t in time_points])

decayRate = np.array([20*np.exp(-.05*t) for t in time_points])

plt.plot(time_points, trajectory[:, 0], label = "pitch")
plt.plot(time_points, trajectory[:, 1], label = "pitch rate")
plt.plot(time_points, decayRate, label="decayRate")
plt.show()

