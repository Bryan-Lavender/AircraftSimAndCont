"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from python_tools.quit_listener import QuitListener
import pyqtgraph as pg
import keyboard
import parameters.simulation_parameters as SIM
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from message_types.msg_delta import MsgDelta
from mystuff.trim import compute_trim
from models.compute_models import compute_ss_model
import numpy as np
import matplotlib.pyplot as plt
from tools.signals import Signals

###
#quitter = QuitListener()

VIDEO = False
PLOTS = True
ANIMATION = True
SAVE_PLOT_IMAGE = False

if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap4_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

#initialize the visualization
if ANIMATION or PLOTS:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if ANIMATION:
    mav_view = MavViewer(app=app)  # initialize the mav viewer
if PLOTS:
    # initialize view of data plots
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()


# create initialization paramters
Va0 = 30.0
alpha0=0.
beta0=0.
mav.initialize_velocity(Va0, alpha0, beta0)



alpha, elevator, throttle = compute_trim(mav, delta)
mav.initialize_velocity(Va0, alpha, beta0)
delta.elevator = elevator
delta.throttle = throttle


trim_state = mav._state[:13]
trim_input = MsgDelta(elevator=elevator,
                        aileron=0,
                        rudder=0,
                        throttle=throttle)
Alon, Blon, Alat, Blat = compute_ss_model(mav,trim_state,trim_input)

eigs = np.linalg.eigvals(Alon)
print(eigs)
x = [ele.real for ele in eigs] 
y = [ele.imag for ele in eigs] 

print(x,y)
plt.scatter(x,y)
for r,i in zip(x,y):
    w = np.sqrt(r**2+i**2)
    print("freq: ", w, " damping: ",-r/w)
#plt.show()
#exit()
sim_time = SIM.start_time
plot_time = sim_time
end_time = 100
#exit()
# main simulation loop
print("Press 'Esc' to exit...")

input_signal = Signals(amplitude=0.3,
                       duration=0.3,
                       start_time=5.0)
while sim_time < end_time:
    delta.elevator = elevator
    delta.aileron = 0
    delta.rudder = 0
    delta.throttle = throttle# 0.6768
    # # ------- set control surfaces -------------
    # if abs((sim_time-3.)) < .01:
    #     delta.elevator += .1
    # if keyboard.is_pressed('w'):
    #     delta.elevator += 0.01  # Adjust value as needed
    # if keyboard.is_pressed('s'):
    #     delta.elevator -= 0.01
    # if keyboard.is_pressed('a'):
    #     delta.rudder -= 0.01
    # if keyboard.is_pressed('d'):
    #     delta.rudder += 0.01
    # if keyboard.is_pressed('q'):
    #     delta.aileron -= 0.01
    # if keyboard.is_pressed('e'):
    #     delta.aileron += 0.01
    # if keyboard.is_pressed('b'):
    #     delta.throttle += 0.01
    # if keyboard.is_pressed('n'):
    #     delta.throttle -= 0.01

    delta.elevator = elevator + input_signal.impulse(sim_time)

    # ------- physical system -------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                            None,  # estimated states
                            None,  # commanded states
                            delta)  # inputs to aircraft
    if ANIMATION or PLOTS:
        app.processEvents()
    if VIDEO is True:
        video.update(sim_time)
        
    # # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation


if SAVE_PLOT_IMAGE:
    data_view.save_plot_image("ch4_plot")

if VIDEO is True:
    video.close()