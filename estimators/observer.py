"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors


class Observer:
    def __init__(self, ts_control, initial_measurements = MsgSensors()):
        # initialized estimated state message
        self.estimated_state = MsgState()
        # use alpha filters to low pass filter gyros and accels
        # alpha = Ts/(Ts + tau) where tau is the LPF time constant

        self.tau = 1
        self.alpha_gyro = .9
        self.alpha_accel = .7
        self.alt_mult = 11
        ##### TODO #####
        self.lpf_gyro_x = AlphaFilter(alpha=self.alpha_gyro, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=self.alpha_gyro, y0=initial_measurements.gyro_y) 
        self.lpf_gyro_z = AlphaFilter(alpha=self.alpha_gyro, y0=initial_measurements.gyro_z) 
        self.lpf_accel_x = AlphaFilter(alpha=self.alpha_accel, y0=initial_measurements.accel_x) 
        self.lpf_accel_y = AlphaFilter(alpha=self.alpha_accel, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=self.alpha_accel, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.98, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.7, y0=initial_measurements.diff_pressure)
        # ekf for phi and theta
        self.attitude_ekf = EkfAttitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()

    def update(self, measurement:MsgSensors, true_state):
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) #+ SENSOR.gyro_x_bias/2
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y) #+ SENSOR.gyro_y_bias/2
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z) #+ SENSOR.gyro_z_bias/2

        # invert sensor model to get altitude and airspeed
        
        self.estimated_state.altitude = self.lpf_abs.update(measurement.abs_pressure)/(1.2682*9.8) 
        self.estimated_state.Va = np.sqrt(self.lpf_diff.update(measurement.diff_pressure)*2/(1.2682))-.08
        print(self.estimated_state.altitude)
        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(measurement, self.estimated_state)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(measurement, self.estimated_state)

        # not estimating these
        self.estimated_state.alpha = true_state.alpha
        self.estimated_state.beta = true_state.beta
        self.estimated_state.gamma = true_state.gamma
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        self.estimated_state.altitude = true_state.altitude
        return self.estimated_state


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        ##### TODO #####
        self.y = self.alpha * self.y + (1 - self.alpha) * u
        return self.y


class EkfAttitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        ##### TODO #####
        self.Q = np.diag([.03, .03])
       
        self.Q_gyro = np.diag([0, 0, 0])
        self.R_accel = np.diag([SENSOR.gyro_sigma, SENSOR.gyro_sigma, SENSOR.gyro_sigma])
        self.N = 1  # number of prediction step per sample
        self.xhat = np.array([[0.0], [0.0]]) # initial state: phi, theta
        self.P = np.diag([0, 0])
        self.Ts = SIM.ts_control/self.N
        self.gate_threshold = 0 #stats.chi2.isf(q=?, df=?)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0) 
        state.theta = self.xhat.item(1) + 0.05

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        ##### TODO #####
        phi = x[0,0]
        theta = x[1,0]

        p = state.p
        q = state.q
        r = state.r


        f_ = np.zeros((2,1))
        f_[0,0] = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        f_[1,0] = q * np.cos(phi) - r * np.sin(phi) 
        return f_

    def h(self, x, measurement, state):
        # measurement model y
        ##### TODO #####
        h_ = np.array([[0],    # x-accel
                        [0],   # y-accel
                        [0]])  # z-accel
        

        phi = x[0,0]
        theta = x[1,0]

        p = state.p
        q = state.q
        r = state.r
        Va = state.Va

        h_[0,0] = q * Va * np.sin(theta) + 9.8 * np.sin(theta) + np.random.normal(loc = 0, scale = SENSOR.gyro_sigma)
        h_[1,0] = r * Va * np.cos(theta) - p * Va * np.sin(theta) - 9.8 * np.cos(theta) * np.sin(phi) + np.random.normal(loc = 0, scale = SENSOR.gyro_sigma)
        h_[2,0] = -q * Va * np.cos(theta) - 9.8 * np.cos(theta) * np.cos(phi) + np.random.normal(loc = 0, scale = SENSOR.gyro_sigma)
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        ##### TODO #####
        Tout = self.Ts
        for i in range(0, self.N):
            Tp = Tout
            self.xhat += Tp * self.f(self.xhat, measurement, state)
            A = jacobian(self.f, self.xhat, measurement, state)
            Ad = np.eye(A.shape[0]) + A*Tp  + A@A*Tp**2
            self.P = Ad@self.P@Ad.T + Tp**2*self.Q
        
    def measurement_update(self, measurement, state):
        # measurement updates
        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T

        ##### TODO #####
        
        S_inv = np.linalg.inv(self.R_accel + C@ self.P @ C.T)
        if stats.chi2.sf((y-h).T @ S_inv @ (y-h), df = 3) > 0.01:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(2) - L@C
            self.P = tmp@self.P@tmp.T + L@self.R_accel@L.T
            self.xhat = self.xhat + L@(y-h)
            

        # if (y-h).T @ S_inv @ (y-h) < self.gate_threshold:
        #     self.P = np.zeros((2,2))
        #     self.xhat = np.zeros((2,1))


class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi, wn, we, psi
    def __init__(self):
        position_Q = .9
        vg_Q = .7
        chi_Q = .2
    

        self.Q = np.diag([
                    position_Q,  # pn
                    position_Q,  # pe
                    vg_Q,  # Vg
                    chi_Q, # chi
                    0.1, # wn
                    0.1, # we
                    0.1, #0.0001, # psi
                    ])
        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma,  # y_gps_n
                    SENSOR.gps_e_sigma,  # y_gps_e
                    SENSOR.gps_Vg_sigma,  # y_gps_Vg
                    SENSOR.gps_course_sigma  # y_gps_course
                    ])
        self.R_pseudo = np.diag([
                    0.1,  # pseudo measurement #1
                    0.1,  # pseudo measurement #2
                    ])
        self.N = 1  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([[0.0], [0.0], [25.0], [0.0], [0.0], [0.0], [0.0]])
        self.P = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.gps_n_old = 0
        self.gps_e_old = 0
        self.gps_Vg_old = 25.
        self.gps_course_old = 0
        self.pseudo_threshold = 0 #stats.chi2.isf(q=?, df=?)
        self.gps_threshold = 100000 # don't gate GPS

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.east = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        f_ = np.array([[0],
                       [0],
                       [0],
                       [0],
                       [0.0],
                       [0.0],
                       [0],
                       ])
        pn, pe, Vg, chi, wn, we, psi = x[:,0]
        Va = state.Va
        g = 9.8
        p = state.p
        q = state.q
        r = state.r
        phi = state.phi
        theta = state.theta
        psi_dot = (q * np.sin(phi)/(np.cos(theta))) + (r * np.cos(phi)/(np.cos(theta)))

        f_[0,0] = Vg * np.cos(chi)
        f_[1,0] = Vg * np.sin(chi)
        f_[2,0] = (Va*np.cos(psi) * (-Va*psi_dot*np.sin(psi)) + Va*np.sin(psi) * (Va*psi_dot*np.cos(psi)))/Vg
        f_[3,0] = (g/Vg)*np.tan(phi)*np.cos(chi-psi)
        f_[4,0] = 0.0
        f_[5,0] = 0.0
        f_[6,0] = psi_dot
        

        return f_

    def h_gps(self, x, measurement, state):
        # measurement model for gps measurements
        h_ = np.array([
            [0.], #pn
            [0.], #pe
            [0.], #Vg
            [0.], #chi
        ])
        pn, pe, Vg, chi, wn, we, psi = x[:,0]
        Va = state.Va
        g = 9.8
        p = state.p
        q = state.q
        r = state.r
        phi = state.phi
        theta = state.theta

        h_[0,0] = pn
        h_[1,0] = pe
        h_[2,0] = Vg
        h_[3,0] = chi
        
        return h_

    def h_pseudo(self, x, measurement, state):
        # measurement model for wind triangale pseudo measurement
        h_ = np.array([
            [0],  # wind triangle x
            [0],  # wind triangle y
        ])
        pn, pe, Vg, chi, wn, we, psi = x[:,0]
        Va = state.Va
        g = 9.8
        p = state.p
        q = state.q
        r = state.r
        phi = state.phi
        theta = state.theta
        Vg = measurement.gps_Vg
        h_[0,0] = (Va * np.cos(psi) - Vg * np.cos(chi))
        h_[1,0] = (Va * np.sin(psi) - Vg * np.sin(chi))
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        Tout = self.Ts
        for i in range(0, self.N):
            Td = Tout
            # propagate model
            self.xhat +=  Td*self.f(self.xhat, measurement, state)

            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)
            # convert to discrete time models
            Ad = np.eye(A.shape[0]) + A*Td + A@A*Td**2
            # update P with discrete time model
            self.P = Ad@self.P@Ad.T + Td**2*self.Q

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, measurement, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.array([[0, 0]]).T
        S_inv = np.linalg.inv(self.R_pseudo + C@self.P@C.T)
        if (y-h).T @ S_inv @ (y-h) < self.pseudo_threshold:
            L = self.P@C.T@S_inv
            tmp = np.eye(2)-L@C
            self.P = tmp@self.P@tmp.T + L@self.R_gps@L.T
            self.xhat = self.xhat + L@(y-h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, measurement, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            y_chi = wrap(measurement.gps_course, h[3, 0])
            y = np.array([[measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi]]).T
            S_inv = np.linalg.inv(self.R_gps + C@self.P@C.T)
            if (y-h).T @ S_inv @ (y-h) < self.gps_threshold:
                L = self.P@C.T@S_inv
                tmp = np.eye(7)-L@C
                self.P = tmp@self.P@tmp.T + L@self.R_gps@L.T
                self.xhat = self.xhat + L@(y-h)

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J