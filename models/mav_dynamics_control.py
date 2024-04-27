"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from message_types.msg_sensors import MsgSensors

import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation, euler_to_quaternion


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self.initialize_velocity(MAV.u0, 0.,0.)
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.
        # update velocity data and forces and moments
        #self._forces_moments = np.array([0.,0.,0.,0.,0.,0.,])


        self.PriorAlt = []
        self.PriorLim = 0

        self.current_forces_moments = np.array([0.,0.,0.,0.,0.,0.,])
    def initialize_velocity(self, va, alpha, beta):
        self._Va = va
        self._alpha = alpha
        self._beta = beta
        self._state[3] = va * np.cos(alpha)*np.cos(beta)
        self._state[4] = va*np.sin(beta)
        self._state[5] = va*np.sin(alpha)*np.cos(beta)
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()

    def calculate_trim_output(self, x):
        alpha, elevator, throttle = x
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self._state[6:10] = euler_to_quaternion(phi, alpha, psi)
        self.initialize_velocity(self._Va, alpha, self._beta)
        delta = MsgDelta()
        delta.elevator = elevator
        delta.throttle = throttle
        forces = self._forces_moments(delta = delta)
       
        return(forces[0]**2 + forces[2]**2 + forces[4]**2)
    ###################################
    # public functions
    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
        #gyro_x = p + n_biax_x equation 7.5


        # simulate rate gyros(units are rad / sec)
        sensor_noise = np.radians(.5)
        self._sensors.gyro_x = self.true_state.p + np.random.normal(0.0, sensor_noise)
        self._sensors.gyro_y = self.true_state.q + np.random.normal(0.0, sensor_noise)
        self._sensors.gyro_z = self.true_state.r + np.random.normal(0.0, sensor_noise)


        # simulate accelerometers(units of g) Equaiton 7.3
        self._sensors.accel_x = self.current_forces_moments[0][0] / MAV.mass + MAV.gravity * np.sin(self.true_state.theta) + np.random.normal(0,SENSOR.accel_sigma)
        self._sensors.accel_y = self.current_forces_moments[1][0] / MAV.mass - MAV.gravity * np.cos(self.true_state.theta)*np.sin(self.true_state.phi) + np.random.normal(0,SENSOR.accel_sigma)
        self._sensors.accel_z = self.current_forces_moments[2][0] / MAV.mass - MAV.gravity * np.cos(self.true_state.theta)*np.cos(self.true_state.phi) + np.random.normal(0,SENSOR.accel_sigma)

        # simulate magnetometers
        # magnetic field in provo has magnetic declination of 12.5 degrees
        # and magnetic inclination of 66 degrees
        dec = np.radians(12.5)
        inc = np.radians(66)
        
        
        #note: np works in radians, and to get rotation we just need to rotate inclination then declination
        R_inc = np.array([
                 [np.cos(-inc), 0, -np.sin(-inc)],
                 [0,            1,             0],
                 [np.sin(-inc), 0,  np.cos(-inc)]
                 ])
        R_dec = np.array([[ np.cos(dec), np.sin(dec), 0],
                 [-np.sin(dec), np.cos(dec), 0],
                 [0           ,0           , 1]])


        mi = (R_inc @ R_dec).T @ np.array([1,0,0])
        phi, theta, psi = [self.true_state.phi, self.true_state.theta, self.true_state.psi]
        mb = euler_to_rotation(phi, theta, psi).T @ mi \
            + np.random.normal(scale = SENSOR.mag_sigma, size =3)
        

        ct = np.cos(self.true_state.theta)
        st = np.sin(self.true_state.theta)
        cp = np.cos(self.true_state.phi)
        sp = np.sin(self.true_state.phi)
        R_v1_b = np.array([
                  [ct, st*sp,  st*cp],
                  [0 ,    cp,    -sp],
                  [-st, ct*sp, ct*cp]
                  ])
        mv1 = R_v1_b @ mb
        self._sensors.mag_x = mb[0]
        self._sensors.mag_y = mb[1]
        self._sensors.mag_z = mb[2]
        print(self._sensors.mag_x, self._sensors.mag_y, self._sensors.mag_z)
        print("MV1: ", mv1, "heading: ", -np.arctan2(mv1[1], mv1[0]) + dec)
        # simulate pressure sensors
        P0 = 29.92
        T0 = 288.15
        L0 =-0.0065
        g = 9.8665
        R = 8.31432
        M = 0.0289699

        hmsg = self.true_state.altitude
        noise_abs_press = np.random.normal(SENSOR.abs_pres_sigma)
        noise_diff_press = np.random.normal(SENSOR.diff_pres_sigma)

        #begin running avgs
        if self.PriorLim <= 1:
            hmsg = hmsg
        else:
            if self.PriorAlt == []:
                hmsg = hmsg
                self.PriorAlt.append(hmsg)
            elif len(self.PriorAlt) < self.PriorLim - 1:
                hmsg = (sum(self.PriorAlt) + hmsg) / (len(self.PriorAlt) + 1)
                self.PriorAlt.append(self.true_state.altitude)
            elif len(self.PriorAlt) == self.PriorLim - 1:
                hmsg = (sum(self.PriorAlt) + hmsg) / (len(self.PriorAlt) + 1)
                self.PriorAlt.append(self.true_state.altitude)
                self.PriorAlt.pop(0)
            
        P = P0 * (1-L0 * hmsg/T0)**((g*M)/(R*L0))
        pground = 130000/3385
        self._sensors.abs_pressure = pground-P \
            #+ noise_abs_press
        self._sensors.diff_pressure = .5 * MAV.rho * self.true_state.Va**2 + noise_diff_press
        

        # simulate GPS sensor
        exp = np.exp(-SENSOR.gps_k * SENSOR.ts_gps)
        
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = exp * self._gps_eta_n + SENSOR.ts_gps * np.random.normal(SENSOR.gps_n_sigma)
            self._gps_eta_e = exp * self._gps_eta_e + SENSOR.ts_gps * np.random.normal(SENSOR.gps_e_sigma)
            self._gps_eta_h = exp * self._gps_eta_h + SENSOR.ts_gps * np.random.normal(SENSOR.gps_h_sigma)
            self._sensors.gps_n = self.true_state.north    + self._gps_eta_n
            self._sensors.gps_e = self.true_state.east     + self._gps_eta_e
            self._sensors.gps_h = self.true_state.altitude + self._gps_eta_h
            self._sensors.gps_Vg = np.sqrt((self.true_state.Va * np.cos(self.true_state.psi) + self.true_state.wn)**2 + (self.true_state.Va * np.sin(self.true_state.psi) + self.true_state.we)**2) + np.random.normal(0,SENSOR.gps_Vg_sigma)
            self._sensors.gps_course = np.arctan2(self.true_state.Va * np.sin(self.true_state.psi) + self.true_state.we, self.true_state.Va * np.cos(self.true_state.psi) + self.true_state.wn) + np.random.normal(0,SENSOR.gps_course_sigma)
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        ##### TODO #####
        Vg_b = self._state[3:6]
        
        # convert wind vector from world to body frame (self._wind = ?)
        # phi, theta, psi = quaternion_to_euler(self._state[6:10])

        # self._wind = euler_to_rotation(phi, theta, psi).T @ (steady_state+gust)
        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        Va_b = Vg_b - steady_state-gust
        ur, vr, wr = Va_b[:,0]

        # compute airspeed (self._Va = ?)

        self._Va = np.linalg.norm(Va_b, axis = 0)[0]
        self._alpha = np.arctan2(wr,ur)
        self._beta = np.arcsin(vr/self._Va)

        # compute angle of attack (self._alpha = ?)
        
        # compute sideslip angle (self._beta = ?)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        phi,theta,psi = quaternion_to_euler(self._state[6:10])
        e0, ex, ey, ez = self._state[6:10]
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        fg_b = MAV.gravity * MAV.mass * np.array([2*(ex*ez - ey*e0), 2*(ey*ez+ex*e0), ez**2+e0**2-ex**2-ey**2])
        
        M_minus = np.exp(-MAV.M * (self._alpha - MAV.alpha0))
        M_plus = np.exp(MAV.M*(self._alpha + MAV.alpha0))
        sigmoid = (1 + M_minus + M_plus)/((1 +M_minus)*(1+M_plus))

        CL = (1-sigmoid)*(MAV.C_L_0 + MAV.C_L_alpha * self._alpha) + sigmoid * (2 * np.sign(self._alpha)*np.sin(self._alpha)**2*np.cos(self._alpha))
        CD = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2/(np.pi * MAV.e*MAV.AR)

        q_bar = .5 * MAV.rho * self._Va**2
        if self._Va == 0:
            F_lift = 0
            F_drag =0
        else:
            F_lift = q_bar * MAV.S_wing * (CL + MAV.C_L_delta_e * delta.elevator + MAV.C_L_q * (MAV.c * q/(2*self._Va)))
            F_drag = q_bar * MAV.S_wing * (CD + MAV.C_D_delta_e * delta.elevator + MAV.C_D_q * (MAV.c * q/(2*self._Va)))


        # compute Lift and Drag coefficients (CL, CD)

        # compute Lift and Drag Forces (F_lift, F_drag)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        # compute longitudinal forces in body frame (fx, fz)
        fx, fz = np.array([[np.cos(self._alpha), -np.sin(self._alpha)], [np.sin(self._alpha), np.cos(self._alpha)]]) @ np.array([-F_drag, -F_lift])

        # compute lateral forces in body frame (fy)
        fy = 0
        CM = 0
        My = 0
        Mx = 0
        Mz = 0
        if self._Va != 0:
            fy = q_bar * MAV.S_wing * (MAV.C_Y_0 + MAV.C_Y_beta * self._beta + p*(MAV.C_Y_p * MAV.b)/(2*self._Va) + r*(MAV.C_Y_r * MAV.b)/(2*self._Va) + MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder)
            # compute logitudinal torque in body frame (My)
            CM = MAV.C_m_0 + MAV.C_m_alpha * self._alpha + q * MAV.C_m_q * (MAV.c/(2 * self._Va))
            My = q_bar * MAV.S_wing * MAV.c * (CM + MAV.C_m_delta_e * delta.elevator)
            # compute lateral torques in body frame (Mx, Mz)
            Mx = q_bar * MAV.S_wing * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta * self._beta + p*(MAV.C_ell_p * MAV.b)/(2*self._Va) + r*(MAV.C_ell_r * MAV.b)/(2*self._Va) + MAV.C_ell_delta_a * delta.aileron + MAV.C_ell_delta_r * delta.rudder)
            
            Mz = q_bar * MAV.S_wing * MAV.b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + p*(MAV.C_n_p * MAV.b)/(2*self._Va) + r*(MAV.C_n_r * MAV.b)/(2*self._Va) + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder)
        

        Fx = fx + fg_b[0] + thrust_prop
        Fy = fy + fg_b[1]
        Fz = fz + fg_b[2]

        Ml = Mx - torque_prop
       
        forces_moments = np.array([[Fx[0], Fy[0], Fz[0], Ml, My, Mz]]).T
        self.current_forces_moments = forces_moments
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # # compute thrust and torque due to propeller
        # ##### TODO #####
        # # map delta_t throttle command(0 to 1) into motor input voltage
        # v_in = delta_t * MAV.V_max

        # # Angular speed of propeller (omega_p = ?)
        # a = MAV.rho * (MAV.D_prop**5) * MAV.C_Q0/((2 * np.pi )**2)
        # b = MAV.rho * (MAV.D_prop**4) * MAV.C_Q1/((2 * np.pi )) * Va + (MAV.KQ * MAV.KV)/MAV.R_motor
        # c = MAV.rho * (MAV.D_prop**3) * MAV.C_Q2 * (Va**2) - (MAV.KQ/MAV.R_motor) * v_in + MAV.KQ * MAV.i0

        # omega_p = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        # # J = 2*np.pi*Va/(omega_p*MAV.D_prop)

        # # Ct = MAV.C_T2 * J **2 + MAV.C_T1*J + MAV.C_T0
        # # CQ = MAV.C_Q2 * J **2 + MAV.C_Q1*J + MAV.C_Q0

        # # thrust and torque due to propeller
        
        # thrust_prop = ((MAV.rho*(MAV.D_prop**4) * MAV.C_T0)/(4 * np.pi**2)) * omega_p**2 + ((MAV.rho * MAV.D_prop**3)*MAV.C_T1*Va/(2 * np.pi)) * omega_p + (MAV.rho*MAV.D_prop**2)*MAV.C_T2 * Va**2
        # torque_prop = (((MAV.rho*MAV.D_prop**5)*MAV.C_Q0)/(4 * np.pi**2))*omega_p**2 + ((MAV.rho * (MAV.D_prop**4)*MAV.C_Q1*Va)/(2 * np.pi))*omega_p + (MAV.rho * (MAV.D_prop**3) * MAV.C_Q2)*Va**2
        #Thanks to timmy flavin
        thrust_prop = 1/2*MAV.rho*MAV.S_prop*((MAV.K_motor*delta_t)**2 - Va**2)
        torque_prop = 0
        return thrust_prop, torque_prop
    

    # def _update_true_state(self):
    #     # rewrite this function because we now have more information
    #     phi, theta, psi = quaternion_to_euler(self._state[6:10])
    #     pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
    #     self.true_state.north = self._state.item(0)
    #     self.true_state.east = self._state.item(1)
    #     self.true_state.altitude = -self._state.item(2)

    #     self.true_state.u = self._state.item(3)
    #     self.true_state.v = self._state.item(4)
    #     self.true_state.w = self._state.item(5)

    #     self.true_state.Va = self._Va
    #     self.true_state.alpha = self._alpha
    #     self.true_state.beta = self._beta
    #     self.true_state.phi = phi
    #     self.true_state.theta = theta
    #     self.true_state.psi = psi
    #     self.true_state.Vg = np.linalg.norm(pdot)
        
    #     self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
    #     self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
    #     self.true_state.p = self._state.item(10)
    #     self.true_state.q = self._state.item(11)
    #     self.true_state.r = self._state.item(12)
    #     self.true_state.wn = self._wind.item(0)
    #     self.true_state.we = self._wind.item(1)
    #     self.true_state.bx = 0
    #     self.true_state.by = 0
    #     self.true_state.bz = 0
    #     self.true_state.camera_az = 0
    #     self.true_state.camera_el = 0

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = -np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
        self.true_state.u = self._state.item(3)
        self.true_state.v = self._state.item(4)
        self.true_state.w = self._state.item(5)
