import numpy as np
import matplotlib.pyplot as plt
A = np.array([[1,.2],[0,1]])
C = np.array([[1.,0.],[0.,1.]])
Q = np.array([[0.04,0],[0,0.01]])
R = np.array([[4,0.],[0.,.04]])
def f(x, noise = False):
    if noise:
        return A@x + np.array([np.random.normal(0,0.04), np.random.normal(0,0.01)]) 
    return A@x
def h(x, noise = False):
    if noise:
        return C@x + np.array([np.random.normal(0,4), np.random.normal(0,0.04)])
    else:
        return C@x 

dt = .2
num_steps = 100
p_actual = np.zeros(num_steps)
v_actual = np.zeros(num_steps) + .5

v_noise = np.zeros(num_steps)
a_noise = np.random.normal(0, .2, num_steps)

v_noise[0] = .5
for k in range(1, num_steps):
    p_actual[k] = p_actual[k-1] + v_actual[k-1] * dt
    v_noise[k] =  v_actual[k-1] + a_noise[k-1]*dt**2

p_noise = p_actual + np.random.normal(0,2, num_steps)

x = np.array([0, 0])
P = np.array([[0.,0.],[0.0,0.0]])
x_hat = [x]
for i in range(1, 100):
    x_pred = f(x, noise=True)
    P_pred = (A@P@A.T+Q)
    L = P_pred@C.T@np.linalg.inv(R+C@P_pred@C.T)
    y = np.array([p_noise[i], v_noise[i]]) - h(x_pred)
    x = x + L@(np.array([p_noise[i], v_noise[i]]) - h(x, noise = True))
    P = (np.eye(len(x))-L@C) @ P_pred
    x_hat.append(x)

x_hat = np.stack(x_hat)
print(x_hat[:,1])
plt.plot(p_actual)
#plt.plot(v_actual)
#plt.plot(v_noise)
plt.plot(p_noise)
plt.plot(x_hat[:,0])
plt.show()