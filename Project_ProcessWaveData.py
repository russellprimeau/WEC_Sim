import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.fft import ifft

# Create function for converting wave height time series data to a wave energy spectrum


def analyze_spectrum(duration, time_step, frequencies, amplitudes):
    # duration = length of simulated waveform (s)
    # time_step = time disretization (s)
    # frequencies = a vector of angular frequency values for which the spectrum is defined (rad/s)
    # amplitudes = a vector of the wave amplitudes at each component frequency of the spectrum (m), which are a
    # function of the wave power spectral density at each frequency.

    # Create independent and dependent variable for the output
    timedomain = np.arange(0, duration+time_step, time_step)
    waterelev = np.zeros(len(timedomain))

    # Assign a random phase angle to each component frequency
    theta = np.random.uniform(0, 2 * np.pi, len(frequencies))

    # Iterate through each component frequency to add the individual waveforms to the combined waveform representing
    # a time series of the water surface elevation.
    for i in range(len(frequencies)):
        waterelev = waterelev + amplitudes[i]*np.cos(frequencies[i]*timedomain - theta[i])

    spectrum = np.transpose(np.array([timedomain, waterelev]))
    return spectrum


def analyze_series(time_s, waterelev_s):
    # time_s, a vector of the times at which wave height (instantaneous water surface elevation) is recorded)
    # waterelev_s, a vector of the instantaneous water surface elevations
    # Returns an array [frequencies, amplitudes] based on the FFT of the time series

    # Create variables for taking the absolute value and magnitude of the frequency decomposition.
    N_s = len(waterelev_s)
    samples_f = len(waterelev_s)
    timestep_s = time_s[1]-time_s[0]

    # Create a vector of the frequency 'bins' into which the FFT will decompose the simulated wave form (rad/s)
    tf_s = 2*np.pi*np.linspace(0, 1/(2.0*timestep_s), int(N_s/2))

    # Take the FFT of the simulated wave form (a time series).
    zeta_f_s = fft(waterelev_s)
    # Take the frequency magnitudes (absolute values), so that the results can be easily plotted instead of being divided
    # between symmetrical positive and negative components (which are a consequence of applying the FFT to a vector with
    # real variables) and normalize by the number of bins.
    zeta_plotf_s = 2.0/samples_f*np.abs(zeta_f_s[:int(samples_f/2)])

    series = np.transpose(np.array([tf_s, zeta_plotf_s]))
    return series
