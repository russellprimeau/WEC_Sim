import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.fft import ifft
from Project_ProcessWaveData import analyze_series
from Project_ProcessWaveData import analyze_spectrum

# Generate a discrete wave energy spectrum from environmental parameters


def make_JONSWAP_spectrum(U_ten: float, F: float, omega):
    # U_ten = mean value of winds over the oceans (m/s)
    # F = fetch: distance from a lee shore, or length of constant wind velocity (m)
    # omega = (n,1) vector of evenly-spaced discrete wave frequencies expressed in radians/s (2 pi wave frequency in Hz)
    # Outputs an array which will computed to approximate a continuous function,

    # Create internal variables
    # Nominal earth gravity, m/s/s
    g = 9.81
    # Phillips constant (dimensionless)
    alpha = .076 * ((U_ten**2/(F*g))**.22)
    # print("alpha = ", alpha)
    # Peak frequency, rad/s
    omega_p = 22 * (((g**2)/(U_ten * F))**(1/3))
    # print("omega_p = ", omega_p)
    # Experimental constant
    gamma = 3.3
    # Discretization step size
    delta = omega[1]-omega[0]

    spec_density = np.zeros(np.shape(omega))

    # Since sigma is a stepwise function of omega, it is helpful to use a loop to compute each value individually.
    for i in range(np.shape(omega)[0]):
        if omega[i] <= omega_p:
            sigma = .07
        else:
            sigma = .09
        # Since the computation of power spectral density for a particular frequency is algebraically complex,
        # here the enhancement factor gamma^r is evaluated as its own variable for each frequency.
        enhance = gamma**np.exp(-((omega[i]-omega_p)**2)/(2*(sigma**2)*(omega_p**2)))
        # Compute spectral density for a particular component frequency (m*m*s)
        spec_density[i] = alpha*g*g/(omega[i]**5) * np.exp((-5/4) * ((omega_p/omega[i])**4))*enhance

    # Assign output variables according to the assignment: frequency and amplitude of the wave form.
    # To obtain wave amplitude, integrate the wave spectral density piecewise. Note that by
    # convention wave height, the value plotted in the example, is twice the amplitude
    # (i.e. the vertical distance from crest to trough).
    frequencies = omega
    amplitudes = np.sqrt(2*spec_density*delta)

    return frequencies, amplitudes


# Call JONSWAP function on sample input, discretizing the function using a
# small step size to obtain an approximation of a continuous function.
U_ten_c = 10
F_c = 30000
omega_c = np.arange(.001, 4.001, .001)

[frequencies_c, amplitudes_c] = make_JONSWAP_spectrum(U_ten_c, F_c, omega_c)

# Since the calculation of wave amplitude at a particular frequency from a continuous spectrum uses a
# piecewise integration, the wave amplitude value depends on the sampling rate (i.e. step size) used to discretize the
# continuous function. This is the motivation for using power spectral density: the magnitude of the PSD function is
# wave amplitude normalized against frequency, so that it does not depend on sample rate. Therefore, calling the
# JONSWAP function on a vector with a larger step size to approximate a discrete spectrum will result in different
# wave amplitudes which do not agree with the first calculation even for the same sea state parameters.
#
# Instead, to find the wave height at only a few discrete component frequencies for the same sea conditions,
# sample the vector of "continuous" values at some sample rate x (every x^th value).

sample = 0
samplerate = 1
frequencies_d = [0]
amplitudes_d = [0]

# Loop to add every xth value from the continuous function to a list of discrete samples.
while sample < len(frequencies_c):
    frequencies_d.append(frequencies_c[sample])
    amplitudes_d.append(amplitudes_c[sample])
    sample = sample + samplerate


# Convert lists to NumPy arrays for plotting.
frequencies_d = np.array(frequencies_d)
amplitudes_d = np.array(amplitudes_d)

# Plot example output
plt.figure(1)
# Convert from wave amplitude to wave height with a factor of 2, to match variables specified in the prompt (and
# feed into the next task more efficiently). However, based on the values, it looks like the example plot from the
# assignment may be using the dimensionless quantity S_zeta(w) / H_sig*T as the dependent variable, rather than
# spectral density (m*m*s/rad), wave height (m) or wave amplitude (m).
plt.plot(frequencies_c, 2*amplitudes_c, 'k', label='continuous spectrum')
plt.stem(frequencies_d, 2*amplitudes_d, '-r', markerfmt='or', label='discrete spectrum')
plt.xlabel('frequencies, rad/s')
plt.ylabel('wave height, m')
plt.xlim(.2, 3.8)
plt.ylim(-.01, 1.2*2*np.max(amplitudes_c))
plt.title("U_10 = {} m/s, F = {} m".format(U_ten_c, F_c))
plt.legend()
plt.grid()
plt.savefig('Task1_JONSWAP.png')

# Output discrete spectrum to a comma-separated text file
spectrum_out = np.zeros([len(frequencies_d), 2], float)
spectrum_out[:, 0] = frequencies_d
spectrum_out[:, 1] = amplitudes_d
np.savetxt('spectrum.csv', spectrum_out, delimiter=',')

######################################################################################################################

# Testing

# Test that function works on data from an external file:
# Read in data from a .txt file specifying a discrete wave spectrum
spectrum_file = 'spectrum.csv'
spectrum_in = np.loadtxt(spectrum_file, delimiter=',')
duration_t = 500
time_step_t = .01
num_freq_t = 800
timeseries_out = analyze_spectrum(duration_t, time_step_t, spectrum_in[:, 0], spectrum_in[:, 1])
np.savetxt('timeseries.csv', timeseries_out, delimiter=',')

# Plot example output
plt.figure(2)
plt.plot(timeseries_out[:, 0], timeseries_out[:, 1], label='synthesized waveform')
plt.xlabel('time, s')
plt.ylabel('water surface elevation, m')
plt.xlim(0, duration_t)
# plt.ylim(-5, 5)
plt.legend()
plt.grid()
plt.savefig('timeseriesplot.png')

#####################################################################################################################


dataspectrum = analyze_series(timeseries_out[:, 0], timeseries_out[:, 1])


plt.figure(3)
# Plot Fourier decomposition of the simulated waveform against angular frequency, with amplitude doubled to show
# wave height
plt.plot(dataspectrum[:, 0], dataspectrum[:, 1], 'r', label='FFT continuous')
plt.xlabel('frequencies, rad/s')
plt.ylabel('wave amplitude, m')
plt.xlim(0, 4.5)
plt.ylim(-.01, 1.2 * np.max(dataspectrum[:, 1]))
plt.legend()
plt.grid()
plt.savefig('DataSpectrum.png')