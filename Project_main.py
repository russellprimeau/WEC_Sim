import argparse
import configparser
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from Project_wave_class import WEC_simulator
from Project_ProcessWaveData import analyze_series
from Project_ProcessWaveData import analyze_spectrum

########################################################################################################################
# M1. Use the argparse and configparser modules to read in data from a configuration file and wave data file,
# which specify initial values for all simulation parameters

# Declare attributes in the parser object for file names and the specifier for the wave data type
parser = argparse.ArgumentParser(description='Read in a simulation configuration file and a wave data file')
parser.add_argument('configfile', metavar='CONFIGURATION', nargs=1, help='Simulation configuration file')
parser.add_argument('wavefile', metavar='WAVEDATA', nargs=1, help='Wave data, either time series [time (s), '
                                                                  'surface elevation (m)] or spectrum [frequency (rad/s),'
                                                                  'amplitude (m)]')
parser.add_argument('wavetype', metavar='WAVETYPE', nargs=1, choices=['time', 'spectrum'], help='Specify wave data '
                                                                                                'type: time series (t) '
                                                                                                'or spectrum (s)')
parser.add_argument('suppressanim', nargs='?', choices=['s', 'leave blank'], default=['leave blank'], help=
"Option argument 's' will suppress animation so the program proceeds directly to generating results using the input from the configuration file")

# Assign values to the parser object when this script is called from the command line
(args) = parser.parse_args()

########################################################################################################################
# M2.Process configuration file

# Open the configuration file specified in the parser and create a dictionary object out of its contents
config = configparser.ConfigParser()
config.read_file(open(args.configfile[0]))
conf = dict(config.items('wave_sim_parameters'))

# Assign values from the dictionary created by the parsing of the configuration file into local variables
# All dictionary objects are evaluated before being assigned a type, to allow input of mathematical expressions

# Buoy simulation parameters
area = float(eval(conf['area']))  # m^2, area of the box parallel to the sea surface elevation (x-y plane)
height = float(eval(conf['height']))  # m, height of the object perpendicular to the cross-sectional area
# Wave height greater than the height of the box plus the body position implies that the object is fully submerged
rho = float(eval(conf['rho']))  # kg/m^3, mass density of the fluid
g = float(eval(conf['g']))  # m/s^2, absolute value of local gravitational acceleration
m_a = float(eval(conf['m_a']))  # kg, mass of exterior body
m_b = float(eval(conf['m_b']))  # kg, mass of internal weight
k_spring = float(eval(conf['k_spring']))  # N/m, spring constant
eq_len = float(eval(conf['eq_len']))  # m, equilibrium distance at which the spring exerts no force
c_damper = float(eval(conf['c_damper']))  # N*s/m, damping constant for energy extraction unit
v_damper = float(eval(conf['v_damper']))  # N*s/m, damping constant for viscous friction and surface tension with water

# Simulation parameters
wave_steps = int(eval(conf['wave_steps']))  # Number of pieces into which the spatial range will be discretized
t_start = float(eval(conf['t_start']))  # start time
period = float(eval(conf['period']))  # s, duration of the simulation/animation
dt = float(eval(conf['dt']))  # s, time step for next frame in animation
max_len = float(eval(conf['max_len']))  # s, maximum number of frames in animation
min_k = float(eval(conf['min_k']))  # N/m, minimum spring constant on slider
max_k = float(eval(conf['max_k']))  # N/m, maximum spring constant on slider
min_damp = float(eval(conf['min_damp']))  # Ns/m, minimum damper constant on slider
max_damp = float(eval(conf['max_damp']))  # Ns/m, maximum damper constant on slider
min_m_a = float(eval(conf['min_m_a']))  # kg, minimum external mass m_a on slider
max_m_a = float(eval(conf['max_m_a']))  # kg, maximum external mass m_a on slider
min_m_b = float(eval(conf['min_m_b']))  # kg, minimum internal mass m_b on slider
max_m_b = float(eval(conf['max_m_b']))  # kg, maximum internal mass m_b on slider
x_range = int(eval(conf['x_range']))  # index of the time step which is the +/- limit of the x-range shown in animation
wave_plot_name = str(conf['wave_plot_name'])  # file name to which plot of wave data is written
wec_plot_name = str(conf['wec_plot_name'])  # file name to which plot of simulation output is written
record_name = str(conf['record_name'])  # file name to which tabulation of simulation output is written
summary_name = str(conf['summary_name'])  # file name to which summary statistics and updated parameters is
# written


# Check that input parameters make sense, and flag any errors. For instance, the buoy must float, so its density must
# be less than that of the water. This can be corrected in the animation GUI, so the program is not terminated.
density_b = (m_a + m_b)/(height*area)
if density_b > rho:
    print("The design parameters specify a buoy with average density {} kg/m^3, which will not float".format(density_b))

########################################################################################################################
# M3. Process wave data: create wave data of whichever type is missing, create and populate an instance of the
# WEC_Simulator-class

# Based on the type of wave data specified in the command line call, generate the data of the missing type
if args.wavetype[0] == 'time':
    wave_timeseries = np.loadtxt(args.wavefile[0], delimiter=",", dtype=float)
    wave_spectrum = analyze_series(wave_timeseries[:, 0], wave_timeseries[:, 1])
    t_start = wave_timeseries[0, 0]
    dt = wave_timeseries[1, 0] - wave_timeseries[0, 0]
    period = wave_timeseries[-1, 0] - wave_timeseries[0, 0]/2

elif args.wavetype[0] == 'spectrum':
    wave_spectrum = np.loadtxt(args.wavefile[0], delimiter=",", dtype=float)
    wave_timeseries = analyze_spectrum(period, dt, wave_spectrum[:, 0], wave_spectrum[:, 1])
else:
    print('Error: wave data type not specified')
    wave_spectrum = np.array(0)  # This code serves only to eliminate an error flag in PyCharm
    wave_timeseries = np.array(0)

# Set the initial position of the buoy to avoid a large transient effect:
# half-submerged, with the spring at the equilibrium length (and assume velocity of both masses = 0)

init_m_a = wave_timeseries[0, 1] - height/2  # Initial position of the bottom surface of the buoy
init_m_b = init_m_a + eq_len  # Initial position of the internal mass, such that the spring is at equilibrium extension

# Create an object of the "wave_sim" class, which condenses the code used in Assignment 2 to simulate a WEC
sim = WEC_simulator()

# Store the configuration parameters in the WEC_simulator class.
sim.set_para(area, height, rho, g, m_a, m_b, k_spring, eq_len, c_damper, v_damper, init_m_a, init_m_b, wave_steps,
             t_start, period, dt, max_len, min_k, max_k, min_damp, max_damp, min_m_a, max_m_a, min_m_b, max_m_b,
             x_range, wave_timeseries, wave_spectrum, wave_plot_name, wec_plot_name, record_name, summary_name)

########################################################################################################################
# M4. Unless suppressed by the user at the command line, animate the simulated waveform-buoy interaction, using
# parameters from the configuration and wave files, while allowing the user to adjust key buoy parameters in real time

if args.suppressanim[0] != 's':
    # Create and configure the plot space for displaying the animation
    fig, ax = plt.subplots(figsize=(11, 8))

    # Create plotting variables
    tEnd = period
    x = np.concatenate((np.flip(-1*(sim.wave_timeseries[1:abs(x_range)+1, 0]), 0), \
                        sim.wave_timeseries[0:abs(x_range), 0]), axis=0)
    y = np.concatenate((np.zeros(x_range), sim.wave_timeseries[0:abs(x_range), 1]), axis=0)

    # Set initial conditions and assign the plots to variables
    # The comma behind the variable name makes the variable "live", still able to update the value in response to
    # new information
    line1, = ax.plot(x, y, label='Wave Height')
    line2, = ax.plot(0, sim.saved[0, 0], 'or', label='Buoy Position (m_a)')
    line3, = ax.plot(0, sim.saved[2, 0], 'ok', label='Internal Mass Position (m_b)')
    plt.xlabel('time, s')
    plt.ylabel('Surface Elevation, m')
    fig.legend()


    # Tell matplotlib not to pin the plot range to the output range
    def init():
        ax.set_xlim(-abs(sim.wave_timeseries[x_range, 0]), abs(sim.wave_timeseries[x_range, 0]))
        ax.set_ylim(1.2*min(sim.wave_timeseries[:, 1]), 1.2*max(sim.wave_timeseries[:, 1]))
        return line1, line2, line3,


    def animate(i):
        wave = np.concatenate((sim.wave_timeseries[i-x_range:i, 1] if i > x_range else np.concatenate((np.zeros(x_range-i),\
                sim.wave_timeseries[0:i, 1]), axis=0), sim.wave_timeseries[i:i+x_range, 1]), axis=0)
        sim.saved[:, i+1] = sim.solve_buoy(sim.saved[:, i], sim.wave_timeseries[i, 0], dt, 0, i)
        line1.set_ydata(wave)
        line2.set_ydata(sim.saved[0, i+1])
        line3.set_ydata(sim.saved[2, i+1])
        return line1, line2, line3,
    # Tell the plot to update the position of the wave every timestep when the animation passes in a new frame number, i.
    # Keep returning the updated line variables with a comma, so they stay "live."


    ani = FuncAnimation(fig,
                        func=animate,
                        init_func=init,
                        frames=len(sim.wave_timeseries[:, 0]) + 1,
                        interval=int(dt * max_len),
                        blit=False,
                        repeat=False)

    # Creates an animation by repeatedly calling a plotting function
    # Needs arguments: a figure, configuration of the figure, the content (function) to animate,
    # the number of iterations (frames) in the animation, how long to display each frame, "blit" (something
    # to do with windows graphical configuration), "repeat" (keep the loop running or not).

    # Adjust simulation parameters in real time using sliders and buttons.

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the spring constant.
    axspring = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    spring_slider = Slider(
        ax=axspring,
        label='Spring constant [N/m]',
        valmin=min_k,
        valmax=max_k,
        valinit=k_spring,
    )

    # Make a horizontal slider to control the damping constant.
    axdamp = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    damp_slider = Slider(
        ax=axdamp,
        label='Damper constant [Ns/m]',
        valmin=min_damp,
        valmax=max_damp,
        valinit=c_damper,
    )

    # Make a vertically oriented slider to control the external mass, m_a
    axm_a = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
    m_a_slider = Slider(
        ax=axm_a,
        label="Mass A [kg]",
        valmin=min_m_a,
        valmax=max_m_a,
        valinit=m_a,
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the external mass, m_a
    axm_b = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
    m_b_slider = Slider(
        ax=axm_b,
        label="Mass B [kg]",
        valmin=min_m_b,
        valmax=max_m_b,
        valinit=m_b,
        orientation="vertical"
    )


    # The function to be called anytime a slider's value changes, which updates the value of the parameter defined
    # by the slider
    def update(val):
        sim.k_spring = spring_slider.val
        sim.c_damper = damp_slider.val
        sim.m_a = m_a_slider.val
        sim.m_b = m_b_slider.val
        fig.canvas.draw_idle()


    # Connect the update function to each slider
    spring_slider.on_changed(update)
    damp_slider.on_changed(update)
    m_a_slider.on_changed(update)
    m_b_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to start the simulation.
    start_button = fig.add_axes([0.6, 0.025, 0.1, 0.04])
    button_1 = Button(start_button, 'Start', hovercolor='0.975')


    def start(event):
        ani.resume()


    button_1.on_clicked(start)

    # Create a `matplotlib.widgets.Button` to stop the simulation.
    stop_button = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button_2 = Button(stop_button, 'Stop', hovercolor='0.975')


    def stop(event):
        ani.pause()


    button_2.on_clicked(stop)
    plt.show()

########################################################################################################################
# M5. Calculate solution: since the simulation shown during the animation will contain discontinuities which are not
# physically meaningful where values were changed by sliders, re-run the simulation using the final parameter values
# selected during the animation

# Inform the user of the final values selected
if sim.k_spring != k_spring or sim.c_damper != c_damper or sim.m_a != m_a or sim.m_b != m_b:
    print("Resimulating from t={} with updated buoy parameters k_spring = {} N/m, c_damper = {} Ns/m, m_a = {} kg, "
          "m_b = {} kg".format(sim.wave_timeseries[0, 0], sim.k_spring, sim.c_damper, sim.m_a, sim.m_b))
else:
    print("Completing simulation and saving results...")

sim.saved[:, 0] = sim.previous
j = 1
# while j < 3: test case
while j < len(sim.saved[0, :]):
    sim.saved[:, j] = sim.solve_buoy(sim.saved[:, j-1], sim.wave_timeseries[j-1, 0], dt, 0, j)
    j += 1

########################################################################################################################
# M6. Process results: calculate summary statistics for analysis and generate output files

# Error checking: y_b must always be between y_a and y_a + height, as the model does not account for physics
# such as transfer of momentum in a collision. Prints an error message if results are invalid.
check1 = min(sim.saved[2, :]-sim.saved[0, :])
check2 = max(sim.saved[2, :]-sim.saved[0, :])-sim.height
check3 = max(sim.saved[0, :] - sim.saved[4,:])
if check1 <= 0:
    print("Invalid results: Position of m_b exceeded physical limits (below position of m_a) in at least one time "
          "step, by ", check1, "m. Try increasing spring stiffness, decreasing damping,, or decreasing m_b/m_a.")
elif check2 >= 0:
    print("Invalid results: Position of m_b exceeded physical limits (above top of buoy) in at least one time step, "
          "by ", check2, "m. Try increasing spring stiffness, decreasing damping, or decreasing m_b/m_a.")
elif check3 > 0:
    print("Invalid results: Position of m_a exceeded the surface elevation in at least one time step by,", check3,
            "m, which is not fully accounted for by the physics model in this program. ")

# Save outputs:
# O1. Plot wave height time series against time, and amplitude against discrete frequencies
figW, axsW = plt.subplots(2, 1)
axsW[0].plot(wave_timeseries[:, 0], wave_timeseries[:, 1], label='Time Series')
axsW[0].set_xlabel('time, s')
axsW[0].set_ylabel('sea surface elevation, m')
axsW[0].set_xlim(0, max(wave_timeseries[:, 0]))
axsW[0].set_ylim(-1.1*max(abs(wave_timeseries[:, 1])), 1.1*max(abs(wave_timeseries[:, 1])))
axsW[0].legend(loc='upper right')
axsW[0].grid()

axsW[1].stem(wave_spectrum[:, 0], wave_spectrum[:, 1], '-r', markerfmt='or', label='Component Waves in Discrete Spectrum')
axsW[1].set_xlabel('frequencies, rad/s')
axsW[1].set_ylabel('wave amplitude, m')
axsW[1].set_xlim(0, 5)
axsW[1].set_ylim(-.01, 1.2 * np.max(wave_spectrum[:, 1]))
axsW[1].legend(loc='upper right')
axsW[1].grid()

plt.savefig(sim.wave_plot_name)

# O2. Plot simulated WEC behavior
figWEC, axs = plt.subplots(5, 1)
figWEC.set_figwidth(8)
figWEC.set_figheight(10)

# Plot wave height vs. time
axs[0].plot(sim.wave_timeseries[:, 0], sim.saved[4, :], label='wave height')
axs[0].set_xlim(0, max(sim.wave_timeseries[:, 0]))
# axs[0].set_ylim()
# axs[0].set_xlabel()
axs[0].set_ylabel("height, m")
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plot buoyancy force vs. time
axs[1].plot(sim.wave_timeseries[:, 0], sim.saved[5, :], 'g', label='buoyancy force')
axs[1].set_xlim(0, max(sim.wave_timeseries[:, 0]))
# axs[1].set_ylim()
# axs[1].set_xlabel()
axs[1].set_ylabel("force, N")
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Plot buoy position (mass a) vs. time
axs[2].plot(sim.wave_timeseries[:, 0], sim.saved[0, :], 'k', marker=',', label='m_a')
axs[2].plot(sim.wave_timeseries[:, 0], sim.saved[2, :], 'r', marker=',', label='m_b')
axs[2].set_xlim(0, max(sim.wave_timeseries[:, 0]))
# axs[2].set_ylim()
# axs[2].set_xlabel()
axs[2].set_ylabel("y-position, m")
axs[2].legend(loc='upper right')
axs[2].grid(True)

# Plot differential velocity (mass 2 - mass 1) vs. time
axs[3].plot(sim.wave_timeseries[:, 0], np.abs(sim.saved[1, :]-sim.saved[3, :]), 'm', label='differential velocity')
axs[3].set_xlim(0, max(sim.wave_timeseries[:, 0]))
axs[3].set_ylim(0, 1.2*max(np.abs(sim.saved[1, :]-sim.saved[3, :])))
# axs[3].set_xlabel("time, s")
axs[3].set_ylabel("velocity, m/s")
axs[3].legend(loc='upper right')
axs[3].grid(True)

# Plot instantaneous damper dissipation power vs. time
axs[4].plot(sim.wave_timeseries[:, 0], sim.saved[6, :], 'r', label='power take off')
axs[4].set_xlim(0, max(sim.wave_timeseries[:, 0]))
axs[4].set_ylim(0, 1.2*max(sim.saved[6, :]))
axs[4].set_xlabel("time, s")
axs[4].set_ylabel("Power, W")
axs[4].legend(loc='upper right')
axs[4].grid(True)

plt.savefig(sim.wec_plot_name)

# O3. Write the full simulation output to an external file 'completed_simulation.csv' so that it can be analyzed or
# input in other programs
column0 = 'time (s)'
column1 = 'position of mass a (m)'
column2 = 'velocity of mass a (m/s)'
column3 = 'position of mass b (m)'
column4 = 'velocity of mass b (m/s)'
column5 = 'wave height (m)'
column6 = 'buoyancy force (N)'
column7 = ' in the damper per timestep (J)'
delimiter = ','
csvheader = column0+delimiter+column1+delimiter+column2+delimiter+column3+delimiter+column4+delimiter+column5+\
            delimiter+column6+delimiter+column7
dataout = np.concatenate((np.reshape(sim.wave_timeseries[:, 0], (len(sim.wave_timeseries[:, 0]), 1)),
                          np.transpose(sim.saved)), axis=1)
np.savetxt(sim.record_name, dataout, delimiter=',', header=csvheader)

# O4. Write performance statistics and final value of all parameters to a summary file 'summary.txt'

# Compute statistics for analyzing simulated performance
avg_power = str(np.average(sim.saved[6, int(.1*len(sim.saved[0, :])):-1]))  # After IC transients dissipate
max_power = str(max(sim.saved[6, int(.1*len(sim.saved[0, :])):-1]))  # After IC transients dissipate
a_b_massratio = str(sim.m_a/sim.m_b)
m_b_critfreq = str(np.sqrt(sim.k_spring/sim.m_b))
buoy_density = str((sim.m_a + sim.m_b)/(sim.height*sim.area))

# Write values to a file
open(sim.summary_name, 'w').close()  # Clear data from previous iterations
with open(sim.summary_name, "a") as myfile:
    myfile.write('[summary statistics]\n')
    myfile.write('average power output = ' + avg_power + ' W\n')
    myfile.write('maximum power output= ' + max_power + ' W\n')
    myfile.write('natural frequency sqrt(k/m_b) = ' + m_b_critfreq + ' rad/s\n')
    myfile.write('buoy density = ' + buoy_density + ' kg/m^3' + '\n')
    myfile.write('configuration input file = ' + args.configfile[0] + '\n')
    myfile.write('wave input file = ' + args.wavefile[0] + '\n')
    myfile.write('wave type = ' + args.wavetype[0] + '\n\n')
    myfile.write('[wave_sim_parameters]\n')
    for parameter in conf:
        string = parameter+"="+str(getattr(sim, parameter))
        myfile.write(string + '\n')
