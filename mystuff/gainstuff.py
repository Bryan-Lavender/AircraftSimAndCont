import numpy as np
import control as ctrl

k = 0.8
a = 0.5
num = [k]
den = [1,a]
G = ctrl.TransferFunction(num, den)

sys = ctrl.tf2ss(G)

desired_poles = np.array([-2 + 1j, -2 - 1j])

K = ctrl.acker(sys.A, sys.B, desired_poles)

# Step 5: Validate the result
A_cl = sys.A - np.dot(sys.B, K)
eigenvalues = np.linalg.eigvals(A_cl)

#print('Feedback Gain K:', K)
#print('Eigenvalues of the closed-loop system:', eigenvalues)



import control as ctrl

# Define C(s) coefficients (replace these with your own values)
kp = 1  # Example proportional gain
ki = 1  # Example integral gain
kd = 1  # Example derivative gain
s = ctrl.TransferFunction.s  # Represents 's'

# Example PID controller C(s) = kp + ki/s + kd*s
C = kp + ki/s + kd*s

# Define G(s) coefficients (replace these with your own values)
K = .8  # Example K
a = .5 # Example a

# Transfer function G(s) = K / (s + a)
G = ctrl.TransferFunction([K], [1, a])

# Calculate the combined transfer function
# Closed-loop characteristic equation: 1 = C(s)*G(s)
L = C * G
sys = ctrl.ss(L)
print(ctrl.acker(sys.A, sys.B, desired_poles))
#print(f"Open-loop transfer function (C(s)G(s)):")
#print(L)