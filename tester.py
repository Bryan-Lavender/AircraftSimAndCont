import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


from airplane_sim.first_airplane import airplane_lrh

airplane = airplane_lrh()
print(airplane.max)

A = np.arange(0, 20, 1)
B = np.arange(20, 0, -1)
y = 2*A+5
x = 2*B+5
plt.title("Matplotlib demo")
plt.plot(y)
plt.plot(x)
plt.show()
print("hello world")