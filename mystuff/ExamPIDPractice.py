import matplotlib.pyplot as plt
import numpy as np
class PIDControl:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, Ts=0.01, sigma=0.05, limit=2.0, init_integrator = 0.0, Lower_limit = 0.0, Upper_limit = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Ts = Ts
        self.limit = limit
        self.upper_limit = Upper_limit
        self.lower_limit = Lower_limit
        self.integrator =init_integrator
        self.error_delay_1 = 0.0
        self.error_dot_delay_1 = 0.0
        self.y_dot = 0.0
        self.y_delay_1 = 0.0
        self.y_dot_delay_1 = 0.0
        # gains for differentiator
        self.a1 = (2.0 * sigma - Ts) / (2.0 * sigma + Ts)
        self.a2 = 2.0 / (2.0 * sigma + Ts)

    def update(self, y_ref, y, reset_flag=False):
        if reset_flag is True:
            self.integrator = 0.0
            self.error_delay_1 = 0.0
            self.y_dot = 0.0
            self.y_delay_1 = 0.0
            self.y_dot_delay_1 = 0.0
        # compute the error
        error = y_ref - y
        
        # update the integrator using trapazoidal rul
        self.integrator = self.integrator \
                          + (self.Ts/2) * (error + self.error_delay_1)
        # update the differentiator
        error_dot = self.a1 * self.error_dot_delay_1 \
                         + self.a2 * (error - self.error_delay_1)
        # PID control
        u = self.kp * error \
            + self.ki * self.integrator \
            + self.kd * error_dot
        # saturate PID control at limit
        u_sat = self._saturate(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.ki) > 0.0001:
            self.integrator = self.integrator \
                              + (self.Ts / self.ki) * (u_sat - u)
        # update the delayed variables
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if self.limit != 0.0:
            if u >= self.limit:
                u_sat = self.limit
            elif u <= -self.limit:
                u_sat = -self.limit
            else:
                u_sat = u
        else:
            if u <= self.lower_limit:
                u_sat = self.lower_limit
            elif u >= self.upper_limit:
                u_sat = self.upper_limit
            else:
                u_sat = u
           
        return u_sat
class PID:
    def __init__(self, kp, ki, kd, high=2., low=-2., ts = 1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.elast = 0.0
        self.I = 0
        self.high = high
        self.low = low
        self.ts = ts
    def update(self, y_ref, y):
        P = y_ref - y
        self.I = self.I \
                          + (self.ts/2) * (P)
        D = (P - self.elast)/self.ts
        self.elast = P
        #print(self.kp * P + self.ki * self.I + self.kd * D)
        retuner = self.kp * P + self.ki * self.I + self.kd * D
        satted = self.saturate(retuner)
        if np.abs(self.ki) > 0.0001:
            self.I = self.I + (self.ts / self.ki) * (retuner - satted)
        return satted
    
    def saturate(self, val):
        if val > self.high:
            return self.high
        elif val < self.low:
            return self.low
        return val


pid_controller = PID(.5,0.0001, 0.02, ts = .1)
y_in = -10.
y_ref = 2
updates = []
sigs = []
updates.append(y_in)
sigs.append(y_ref)
for i in range(0, 100):
    if i == 25:
        y_ref = 10.
    if i == 50:
        y_ref = 0.
    if i == 75:
        y_ref = -10.
    
    y_in =y_in +  pid_controller.update(y_ref, y_in)
    
    updates.append(y_in)

    sigs.append(y_ref)

print(sigs)
print(updates)

plt.plot(sigs)
plt.plot(updates)
plt.show()
