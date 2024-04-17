"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from tools.quit_listener import QuitListener
import numpy as np
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot import Autopilot
# from control.autopilot_lqr import Autopilot
# from control.autopilot_tecs import Autopilot
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from message_types.msg_delta import MsgDelta 
from mystuff.trim import do_trim
from viewers.sensor_viewer import SensorViewer

#quitter = QuitListener()

VIDEO = False
PLOTS = True
ANIMATION = True
SAVE_PLOT_IMAGE = False
if ANIMATION:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if True:
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)
if True:
    sensor_view = SensorViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)
# video initialization
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap6_video.avi",
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
delta = do_trim(mav, Va=25, alpha = 0)
autopilot = Autopilot(delta,mav,SIM.ts_simulation)
print(SIM.ts_simulation)
# autopilot commands
from message_types.msg_autopilot import MsgAutopilot
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.05)
altitude_command = Signals(dc_offset=100.0,
                           amplitude=np.radians(45),
                           start_time=5.0,
                           frequency=0.015)
course_command = Signals(dc_offset=0,
                         amplitude=np.radians(45),
                         start_time=5.0,
                         frequency=0.1)

# initialize the simulation time
sim_time = SIM.start_time
end_time = 100


input_signal = Signals(amplitude=0.3,
                       duration=0.3,
                       start_time=4.0)
# main simulation loop
print("Press 'Esc' to exit...")
print(Va_command.square(sim_time), course_command.square(sim_time), altitude_command.square(sim_time))
while sim_time < end_time:
     # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = course_command.square(sim_time)
    commands.altitude_command = altitude_command.square(sim_time)

    commands.airspeed_command = 25.0
    #commands.altitude_command = 100.

    # -------autopilot-------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state = mav.true_state  # uses true states in the control
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics


    # ------- animation -------
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                            None,  # estimated states
                            commanded_state,  # commanded states
                            delta)  # inputs to aircraft
    if ANIMATION or PLOTS:
        app.processEvents()
    if VIDEO is True:
        video.update(sim_time)
        
    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if True:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                            None,  # estimated states
                            commanded_state,  # commanded states
                            delta)  # inputs to aircraft
    if True:
        sensor_view.update(measurements)

    if ANIMATION or DATA_PLOTS or SENSOR_PLOTS:
        app.processEvents()
if SAVE_PLOT_IMAGE:
    data_view.save_plot_image("ch6_plot")

if VIDEO is True:
    video.close()




