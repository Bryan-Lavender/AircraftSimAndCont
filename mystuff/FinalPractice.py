import numpy as np

AOA = np.deg2rad(5)
conv = np.array([[np.cos(AOA), -np.sin(AOA)],[np.sin(AOA), np.cos(AOA)]])
F = np.array([[-2.75625],[-55.125]])

print(conv@F)