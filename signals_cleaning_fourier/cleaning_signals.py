import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

"""
# parameters
amplitude = 1 # micro volts
frequency = 1 # Hz
phase_shift = 0 # radians
sampling_rate = 128 # Fs, tiene que ser del doble de la frecuencia que queremos analizar
time = 10 # seconds
"""

def get_sine_wave(amplitude: float, frequency: float, phase_shift: float, sampling_rate: float, time: float):
    t = np.linspace(0, time, int(sampling_rate * time), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    return t, sine_wave


def filter_signal(order, low, high, sig):
    b, a = signal.butter(order, [low, high], 'band')  # type: ignore
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal


def get_psd(sig):
    f, pxx = signal.welch(sig, 256, nperseg=256)
    db_hz = [10 * np.log10(x) for x in pxx]
    return f, db_hz


def plot_amplitude(x,y):
    fig = plt.figure()
    plt.plot(x,y)
    plt.title("signal")
    plt.xlabel("time (seg)")
    plt.ylabel("amplitude (mv)")
    plt.show()

def subplots_noise_clean(t, comb, clean_signal):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, comb)
    ax1.set_title('Noisy signal')
    ax2.plot(t, clean_signal)
    ax2.set_title('Cleaned signal')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_psd(x, y):
    fig = plt.figure()
    plt.plot(x,y)
    plt.title("Power spectrum density")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("PSD (V**2/Hz)")
    plt.show()

# 1Hz signal
# t_sec, sine = get_sine_wave(amplitude=1, frequency=1, phase_shift=0, sampling_rate=128, time=10)
# plot_amplitude(t_sec, sine)

# 20Hz signal
t_sec, sine1 = get_sine_wave(amplitude=1, frequency=20, phase_shift=0, sampling_rate=256, time=10)
# plot_amplitude(t_sec, sine1)

# 80Hz signal
t_sec, sine2= get_sine_wave(amplitude=1, frequency=80, phase_shift=0, sampling_rate=256, time=10)
# plot_amplitude(t_sec, sine2)

# 120Hz signal
t_sec, sine3 = get_sine_wave(amplitude=1, frequency=120, phase_shift=0, sampling_rate=256, time=10)
# plot_amplitude(t_sec, sine3)

# combined signal (no estacionaria)
comb = sine1+sine2+sine3

# plot_amplitude(t_sec, comb)


# con ésto se ve cuáles son las principales frecuencias de una señal
f,psd = get_psd(comb)
plot_psd(f, psd)


# pasar lo de banda como la frecuencia sobre la frecuencia de muestreo
clean_signal = filter_signal(order=4, low=1/256, high=80/256, sig=comb)
subplots_noise_clean(t_sec, comb, clean_signal)



