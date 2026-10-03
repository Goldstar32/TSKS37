import numpy as np
from matplotlib import pyplot as plt
import scipy
from pydub import AudioSegment
from pydub.playback import play

sample_rate1, audio1 = scipy.io.wavfile.read("CantinaBand3.wav")
sample_rate2, audio2 = scipy.io.wavfile.read("StarWars3.wav")

fft1 = scipy.fft.fft(audio1)
fft2 = scipy.fft.fft(audio2)

freq1 = scipy.fft.fftfreq(len(audio1), d=1/sample_rate1)
freq2 = scipy.fft.fftfreq(len(audio2), d=1/sample_rate2)

# Figure

# Create a Figure object and an Axes object
fig, axs = plt.subplots(2, 2, layout="constrained")

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

# plt.show()

# Put source audios into one matrix
X = np.column_stack((audio1, audio2))

# 0 <= ALPHA <= 1
# ALPHA determines the how loud each channel is.
ALPHA = 0.5

# 0 <= THETA <= pi/2
# THETA determines the spacing between the mixed audios.
THETA = np.pi/4

# If ALPHA is 0.5 and THETA is pi/4, both audios can be heard in both channels.

# Create weight matrix
W = np.array([[ALPHA*np.cos(THETA), (1-ALPHA)*np.sin(THETA)],
              [-ALPHA*np.sin(THETA), (1-ALPHA)*np.cos(THETA)]])

Y = X@W

# Save mixed audio
scipy.io.wavfile.write(
    "mixed_audio.wav",
    sample_rate1,
    Y.astype(np.int16)
)

play(AudioSegment.from_wav("mixed_audio.wav"))
