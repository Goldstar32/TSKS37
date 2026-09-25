import numpy as np
from matplotlib import pyplot as plt
import scipy
from pydub import AudioSegment
from pydub.playback import play

sample_rate1, audio1 = scipy.io.wavfile.read("CantinaBand3.wav")
sample_rate2, audio2 = scipy.io.wavfile.read("StarWars3.wav")

# Figure

# Create a Figure object and an Axes object
fig, axs = plt.subplots(2, 2, layout="constrained")

fft1 = scipy.fft.fft(audio1)
fft2 = scipy.fft.fft(audio2)

freq1 = scipy.fft.fftfreq(len(audio1), d=1/sample_rate1)
freq2 = scipy.fft.fftfreq(len(audio2), d=1/sample_rate2)


# Waveform 1
axs[0, 0].plot(audio1)
axs[0, 0].set_title("Audio 1 - Waveform")
axs[0, 0].set_xlabel("Sample")
axs[0, 0].set_ylabel("Amplitude")
# Max x is sample rate times total time of audio in seconds
axs[0, 0].set_xlim(0, sample_rate1*3)

# Waveform 2
axs[0, 1].plot(audio2)
axs[0, 1].set_title("Audio 2 - Waveform")
axs[0, 1].set_xlabel("Sample")
axs[0, 1].set_ylabel("Amplitude")
# Max x is sample rate times total time of audio in seconds
axs[0, 1].set_xlim(0, sample_rate2*3)

# Spectrum 1
axs[1, 0].plot(freq1, np.abs(fft1))
axs[1, 0].set_title("Audio 1 - Spectrum")
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Magnitude")
# Min/max x is min/max frequency
axs[1, 0].set_xlim(np.min(freq1), np.max(freq1))

# Spectrum 2
axs[1, 1].plot(freq2, np.abs(fft2))
axs[1, 1].set_title("Audio 2 - Spectrum")
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Magnitude")
# Min/max x is min/max frequency
axs[1, 1].set_xlim(np.min(freq2), np.max(freq2))

plt.show()