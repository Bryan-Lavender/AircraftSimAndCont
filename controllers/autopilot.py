"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import numpy as np
import parameters.control_parameters as AP
# from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from controllers.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from controllers.pid_control import PIDControl
from controllers.pd_control import PDControl
airspeed_throttle_kp = 0.0
airspeed_throttle_ki = 0.0001




yaw_damper_kp  = 10.0
yaw_damper_kd = 1.0


####elevator from alpha
alpha_elevator_kp = -5
alpha_elevator_ki = -0.0018
alpha_elevator_kd = -.5

###Alpha from Gamma
gamma_alpha_kp = 50.
gamma_alpha_ki = 0.000001
gamma_alpha_kd = 5
###Alpha Prior Set

###Gamma from Altitude
altitude_gamma_kp = 500.
altitude_gamma_ki = 0.00000018
altitude_gamma_kd = 50

###Delta a from roll
roll_Aileron_kp = 1.
roll_Aileron_ki = 0.0018
roll_Aileron_kd = .1

###roll from chi
chi_roll_kp = 10.
chi_roll_ki = 0.0018
chi_roll_kd = 1

class Autopilot:
    def __init__(self, delta, mav, ts_control):
        # instantiate lateral-directional controllers
        
        
        self.throttle_from_airspeed = PIControl(
            kp = airspeed_throttle_kp,
            ki = airspeed_throttle_ki,
            Ts = ts_control,
            limit = 1.0,
            init_integrator=delta.throttle/airspeed_throttle_ki,
        )

        self.elevator_from_alpha = PIDControl(
            kp = alpha_elevator_kp,
            ki = alpha_elevator_ki,
            kd = alpha_elevator_kd,
            limit = 1.0,
            Ts = ts_control,
            init_integrator=delta.elevator/alpha_elevator_ki

        )
        #### Alpha from gamma
        self.alpha_from_gamma = PIDControl(
            kp = gamma_alpha_kp,
            ki = gamma_alpha_ki,
            kd = gamma_alpha_kd,
            limit = 0.0,
            Ts = ts_control,
            init_integrator=mav._alpha/gamma_alpha_ki,
            Upper_limit=0.20944,
            Lower_limit=-0.0349066
        )
        self.alpha_prior_set = None
        #### gamma from Altitude
        self.gamma_from_altitude = PIDControl(
            kp = altitude_gamma_kp,
            ki = altitude_gamma_ki,
            kd = altitude_gamma_kd,
            limit = 0.0,
            Ts = ts_control,
            init_integrator=0/altitude_gamma_ki,
            Upper_limit=0.261799,
            Lower_limit=-0.261799
        )


        #### Aileron from roll
        self.aileron_from_roll = PIDControl(
            kp = roll_Aileron_kp,
            ki = roll_Aileron_ki,
            kd = roll_Aileron_kd,
            limit = 0.785398,
            Ts = ts_control,
            init_integrator=0/roll_Aileron_ki,
            Upper_limit=0.0,
            Lower_limit=-0.0
        )

        
        #### roll from chi
        self.roll_from_chi = PIDControl(
            kp = chi_roll_kp,
            kd = chi_roll_kd,
            ki = chi_roll_ki,
            limit = 0.785398,
            Ts = ts_control,
            init_integrator= 0/chi_roll_ki,
            Lower_limit=0,
            Upper_limit=0
        )

        self.yaw_damper = PDControl(
            kp = yaw_damper_kp, 
            kd = yaw_damper_kd, 
            Ts = ts_control, 
            limit = 1.0
        )
        self.commanded_state = MsgState()
    def update(self, cmd, state):
        Va_set = cmd.airspeed_command
        Chi_set = cmd.course_command
        Alt_set = cmd.altitude_command
        

        delta = MsgDelta(elevator=0,aileron=0,rudder=0,throttle=0)
        
    #### TODO #####
        # lateral autopilot
        delta.rudder = self.yaw_damper.update(0, state.beta)
        
        # longitudinal autopilot
        delta.throttle = self.throttle_from_airspeed.update(Va_set, state.Va)
        #alpha_set = 0.104
        #alpha_set = self.alpha_from_gamma.update(0.104 ,state.gamma)
        #print(alpha_set)
        delta.elevator = self.elevator_from_alpha.update(self.alpha_from_gamma.update(self.gamma_from_altitude.update(Alt_set, state.altitude) ,-state.gamma), state.alpha)

        delta.aileron = self.aileron_from_roll.update(self.roll_from_chi.update(Chi_set,state.chi), state.phi)

        # construct control outputs and commanded states
        
        self.commanded_state.altitude = 0
        self.commanded_state.Va = 0
        self.commanded_state.phi = 0
        self.commanded_state.theta = 0
        self.commanded_state.chi = 0
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
