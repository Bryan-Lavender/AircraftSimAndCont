"""
mavsim_python
    - Chapter 8 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/21/2019 - RWB
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
from estimators.observer import Observer
# from estimation.observer_full import Observer
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from viewers.sensor_viewer import SensorViewer

from message_types.msg_delta import MsgDelta 
from mystuff.trim import do_trim

import numpy as np
import parameters.simulation_parameters as SIM
import pyqtgraph as pg
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot import Autopilot
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from viewers.sensor_viewer import SensorViewer
from message_types.msg_delta import MsgDelta 
from mystuff.trim import do_trim

#quitter = QuitListener()

VIDEO = False
DATA_PLOTS = True
SENSOR_PLOTS = True
ANIMATION = True
SAVE_PLOT_IMAGE = False
COMPUTE_MODEL = False

# video initialization
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap8_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

#initialize the visualization
if ANIMATION or DATA_PLOTS or SENSOR_PLOTS:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if ANIMATION:
    mav_view = MavViewer(app=app)  # initialize the mav viewer
if DATA_PLOTS:
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)
if SENSOR_PLOTS:
    sensor_view = SensorViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()
delta = do_trim(mav, Va=25, alpha = 0)
autopilot = Autopilot(delta,mav,SIM.ts_simulation)
observer = Observer(SIM.ts_simulation)

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
                         amplitude=np.radians(10),
                         start_time=5.0,
                         frequency=0.1)

# initialize the simulation time
sim_time = SIM.start_time
end_time = 1000
# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:

    # -------autopilot commands-------------
    if sim_time < 10:
        commands.airspeed_command = 25
        commands.course_command = 0.
        commands.altitude_command = 100
    elif sim_time < 40:
        commands.airspeed_command = 30
        commands.course_command = 0.785398
        commands.altitude_command = 300
    else:
        commands.airspeed_command = 25
        commands.course_command = 0.785398
        commands.altitude_command = 200
    # -------- autopilot -------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state_act = mav.true_state
    estimated_state = observer.update(measurements, estimated_state_act)  # estimate states from measurements
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------- physical system -------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------- update viewer -------------
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if DATA_PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         delta)  # inputs to aircraft
    if SENSOR_PLOTS:
        sensor_view.update(measurements)
    if ANIMATION or DATA_PLOTS or SENSOR_PLOTS:
        app.processEvents()
    if VIDEO is True:
        video.update(sim_time)
        
    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation

# Save an Image of the Plot
if SAVE_PLOT_IMAGE:
    if DATA_PLOTS:
        data_view.save_plot_image("ch8_data_plot")
    if SENSOR_PLOTS:
        sensor_view.save_plot_image("ch8_sensor_plot")

if VIDEO is True:
    video.close()




