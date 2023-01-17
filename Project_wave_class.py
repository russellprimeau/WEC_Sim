import numpy as np


class WEC_simulator:
    def __init__(self):
        pass

    def set_para(self, area, height, rho, g, m_a, m_b, k_spring, eq_len, c_damper, v_damper, init_m_a, init_m_b, \
                 wave_steps, t_start, period, dt, max_len, min_k, max_k, min_damp, max_damp, min_m_a, max_m_a,
                 min_m_b, max_m_b, x_range, wave_timeseries, wave_spectrum, wave_plot_name, wec_plot_name, record_name, summary_name):
        self.area = area  # m^2, area of the box parallel to the sea surface elevation (x-y plane)
        self.height = height  # m, height of the object perpendicular to the cross-sectional area, so that wave height
        # greater than the height of the box plus the body position implies that the object is fully submerged
        self.rho = rho  # kg/m^3, mass density of the fluid
        self.g = g  # m/s^2, absolute value of local gravitational acceleration
        self.m_a = m_a  # kg, mass of exterior body
        self.m_b = m_b  # kg, mass of internal weight
        self.k_spring = k_spring  # N/m, spring constant
        self.eq_len = eq_len  # m, equilibrium distance at which the spring exerts no force
        self.c_damper = c_damper  # N*s/m, damping constant for energy extraction unit
        self.v_damper = v_damper  # N*s/m, damping constant for viscous friction and surface tension with water
        self.init_m_a = init_m_a  # m, initial position of the bottom surface of the buoy
        self.init_m_b = init_m_b  # m, initial position of the bottom surface of the internal mass
        self.wave_steps = wave_steps  # Number of pieces into which the spatial range will be discretized
        self.t_start = t_start  # start time
        self.period = period  # s, duration of animation
        self.dt = dt  # s, time step for next frame in animation
        self.max_len = max_len  # s, maximum number of frames in animation
        self.min_k = min_k  # N/m, minimum spring constant on slider
        self.max_k = max_k  # N/m, maximum spring constant on slider
        self.min_damp = min_damp  # Ns/m, minimum damper constant on slider
        self.max_damp = max_damp  # Ns/m, maximum damper constant on slider
        self.min_m_a = min_m_a  # kg, minimum external mass m_a on slider
        self.max_m_a = max_m_a  # kg, maximum external mass m_a on slider
        self.min_m_b = min_m_b  # kg, minimum internal mass m_b on slider
        self.max_m_b = max_m_b  # kg, maximum internal mass m_b on slider
        self.x_range = x_range  # index of the time step which is the +/- limit of the x-range shown in animation
        self.wave_plot_name = wave_plot_name  # file name to which plot of wave data is written
        self.wec_plot_name = wec_plot_name  # file name to which plot of simulation output is written
        self.record_name = record_name  # file name to which tabulation of simulation output is written
        self.summary_name = summary_name  # file name to which summary statistics and updated parameters is written
        self.wave_timeseries = wave_timeseries  # array of time series of wave heights: [time (s), surface height
        # relative to mean sea surface elevation (m)]
        self.wave_spectrum = wave_spectrum  # array of discrete frequencies and wave amplitude at that frequency
        # [frequency (Hz), amplitude (m)]
        self.previous = np.array([init_m_a, 0, init_m_b, 0, 0, 0, 0])  # array of initial conditions for the solution
        self.saved = np.zeros((len(self.previous),len(wave_timeseries[:,0])))
        self.saved[:, 0] = self.previous
        # The vectors "previous" and each element of the array saved[:, i] represent:
        # results[0] = position of mass a (m)
        # results[1] = velocity of mass a (m/s)
        # results[2] = position of mass b (m)
        # results[3] = velocity of mass b (m/s)
        # results[4] = wave height (m)
        # results[5] = buoyancy force (N)
        # results[6] = energy dissipated in the damper per timestep (J)

    def submerged_volume(self, depth: float) -> float:
        # Returns the volume (m^3) of fluid displaced by the submerged portion of an object. Could be redefined or
        # expanded to a class to provide more abstraction. For the purposes of this assignment, it is assumed that the
        # object is a prism, with constant surface area in the plane perpendicular to the wave height.
        #
        # Prism's height is submerged up to depth (m), which cannot exceed the object's height.

        volume = self.area * depth
        return volume

    def compute_buoyancy(self, body_position, wave_height):
        # Returns buoyancy force (N) acting on an object based on its position relative to a fluid free surface,
        # assuming that the length of the object in the direction of the wave propagation is small enough
        # relative to the wavelength that it is reasonable to approximate the wetted surface of the object as being
        # of uniform depth at any point in time (i.e. ignoring variations in the wave height around the perimeter of
        # the object.) Force is oriented in the upward direction.
        #
        # body_position (m) is the instantaneous elevation of the lowest surface of the object relative to the mean
        # sea surface elevation.
        #
        # wave_height (m) is the instantaneous elevation of the free fluid surface relative to the mean sea surface
        # elevation. Note that this differs from the conventional hydrodynamic definition of wave height.
        if wave_height >= (self.height + body_position):  # object is fully submerged, so buoyancy force is constant
            # regardless of depth
            force = self.rho * self.g * self.submerged_volume(self.height)
        elif wave_height < body_position:  # object is fully free of the fluid, and experiences no buoyancy
            force = 0
        else:  # object is partially submerged, buoyancy is a function of shape
            # and depth.
            force = self.rho * self.g * self.submerged_volume(wave_height - body_position)
        return force

    def waveform(self, wave_time: float) -> float:
        # wave_time (s) is the time at which wave height is calculated
        # returns wave height (m), which is water surface elevation above mean surface elevation.
        return np.interp(wave_time, self.wave_timeseries[:,0], self.wave_timeseries[:,1])

    def sys_ODE(self, ODE_x, ODE_time, Y):
        # ODE_time (s) is the time at which to calculate the speeds and accelerations of the two objects,
        # which is used by the function to look up the wave height at that time.
        # y (m, m/s, m, m/s) is the vector of position and speed values at time t that are used to calculate the
        # velocities and accelerations.

        # A is the array of coefficients for the position and velocity values in the vector of input values, Y.
        A = np.zeros((4, 4))
        A[0, 1] = 1
        A[1, :] = [-self.k_spring / self.m_a, -(self.c_damper + self.v_damper) / self.m_a, self.k_spring / self.m_a, \
                   self.c_damper / self.m_a]
        A[2, 3] = 1
        A[3, :] = [self.k_spring / self.m_b, self.c_damper / self.m_b, -self.k_spring / self.m_b, \
                   -self.c_damper / self.m_b]

        # B is the vector of constants added to the YA product term to calculate the differential vector dY/dt.
        # B is not made up of constants in this system, since one of the terms depends on the buoyancy force,
        # with is time variant. By treating it as a constant at each timestep, we are implicitly assuming that the
        # current wave height does not contain any information about the wave height at the next time step, rather than
        # assuming that the differential of the wave height might be used to predict the wave height
        # in the next time step.
        B = np.array([[0], [(self.compute_buoyancy(Y[0], self.waveform(ODE_time)) / self.m_a) - \
                            ((self.k_spring * self.eq_len) / self.m_a) - self.g], [0],
                      [((self.k_spring * self.eq_len) / self.m_b) - self.g]])

        Y = np.reshape(Y,(4,1))
        # Calculating the value of the vector of first derivatives and reshaping to a vector for consistency:
        y_diff = np.reshape(np.dot(A, Y) + B,(4,))
        return y_diff


    def rk_step(self, fun, y_last, rk_time, delta, rk_x):
        # fun is a function which represents a system of first order ODEs
        # y_last an array of the last known set of solutions to the ODE, which are used to solve for the next value
        # rk_time (s) is the time at which the function is numerically solved
        # delta (s) is the interval between times at which the equation is solved (which may vary)
        # adapted from IP501320 lecture notes, 26/09/2022, on Runge-Kutta fourth order method for numerical integration
        k1 = delta * fun(rk_x, rk_time, y_last)
        k2 = delta * fun(rk_x, rk_time + delta / 2, y_last + 0.5 * k1)
        k3 = delta * fun(rk_x, rk_time + delta / 2, y_last + 0.5 * k2)
        k4 = delta * fun(rk_x, rk_time + delta, y_last + k3)
        return y_last + (1/6) * k1 + (2/6) * k2 + (2/6) * k3 + (1/6) * k4

    def solve_buoy(self, prior, solver_time, timestep, solve_x, index):
        # solver_time is the time at which the solution stored in prior is valid (s)
        # timestep is the distance in the time domain between the prior solution and the results being solved for (s)
        # solve_x is the position of the buoy, which is always 0, but is needed to agree with the syntax of a module

        # Declare a vector to write the solution to:
        result = np.zeros((len(prior), 1))
        # Calculate the position and velocities of both masses after a timestep has elapsed from the previous results
        result[0:4, 0] = np.resize(self.rk_step(self.sys_ODE, prior[0:4], solver_time, timestep, solve_x), (4,))
        # Record wave height
        result[4, 0] = self.waveform(solver_time)
        # Record buoyancy force
        result[5, 0] = self.compute_buoyancy(result[0, 0], result[4, 0])
        # Record average rate of energy dissipation in the damper, which is an absolute value function since energy is
        # dissipated regardless of whether the direction of the velocity (up or down). The work done in the damper is
        # equal to the force exerted by the damper multiplied by the differential displacement (the change in the
        # distance between A and B). The force exerted by the damper is equal to the damping constant multiplied by the
        # differential velocity (the difference between the velocity of A and B).
        result[6, 0] = self.c_damper * np.abs((prior[1] - prior[3])) * \
                        np.abs((prior[2] - result[0,0])-(prior[2]- prior[0]))/timestep
        return result[:,0]
