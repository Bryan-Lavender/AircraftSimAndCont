
from message_types.msg_delta import MsgDelta
from models.mav_dynamics_control import MavDynamics
from scipy.optimize import minimize
import numpy as np
def compute_trim(mav: MavDynamics, delta: MsgDelta):
    x0 = [mav._alpha, delta.elevator, delta.throttle]
    bounds = [(0,np.deg2rad(12)), (-1,1), (0,1)]
    res = minimize(mav.calculate_trim_output, x0, bounds = bounds,method='SLSQP')
    return(res.x[0], res.x[1], res.x[2])