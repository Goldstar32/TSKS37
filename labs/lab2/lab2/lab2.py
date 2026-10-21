import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq



#if __name__ == "__main__":
fs1, audio1 = wavfile.read("StarWars3.wav")
fs2, audio2 = wavfile.read("CantinaBand3.wav")

print("StarWars:")
print("Sample rate:", fs1, "Hz")
print("Dimensions:", audio1.shape)
print("\nCantinaBand:")
print("Sample rate:", fs2, "Hz")
print("Dimensions:", audio2.shape)

fft1 = fft(audio1)
fft2 = fft(audio2)
freq1 = fftfreq(n=len(audio1), d=1/fs1)
freq2 = fftfreq(n=len(audio2), d=1/fs2)

figure, ax = plt.subplots(2, 2, figsize=(12, 8))

# Top left is StarWars waveform.
ax[0, 0].plot(audio1)
ax[0, 0].set_title("StarWars")
ax[0, 0].set_xlabel("Sample")
ax[0, 0].set_ylabel("Amplitude")

# Top right is CantinaBand waveform.
ax[0, 1].plot(audio2)
ax[0, 1].set_title("CantinaBand")
ax[0, 1].set_xlabel("Sample")
ax[0, 1].set_ylabel("Amplitude")

# Bottom left is StarWars Fourier transform.
ax[1, 0].plot(freq1, np.abs(fft1))
ax[1, 0].set_title("StarWars Fourier transform")
ax[1, 0].set_xlabel("Frequency (Hz)")
ax[1, 0].set_ylabel("Magnitude")

# Bottom right is CantinaBand Fourier transform.
ax[1, 1].plot(freq2, np.abs(fft2))
ax[1, 1].set_title("CantinaBand Fourier transform")
ax[1, 1].set_xlabel("Frequency (Hz)")
ax[1, 1].set_ylabel("Magnitude")

# Adjust spacing
plt.tight_layout()

# Show the figure
plt.savefig("audio_plots.png")


alpha = 0.05
theta = np.pi/ 2.1

'''
Good reconstruction:
alpha 0.5: rotation
t = pi/4: equal mix/rotation

Bad reconstruction:
alpah = 0.05
t = pi/2.1 (unequal rotation in mixing and demixing)
'''

# Mixing matrix
A = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),   np.cos(theta)]
])
B = np.array([
    [alpha, 0],
    [0, 1-alpha]
])
W = A @ B

# Put the two original audio signals into one matrix
X = np.column_stack((audio1, audio2))

# Mix the signals
Y = X @ W

print("\nMixing matrix:")
print(W)

print("\nMixed matrix dimensions:")
print(Y.shape)


# Save mixed tracks
wavfile.write(
    "mixed1.wav",
    fs1,
    Y[:, 0].astype(np.int16)
)

wavfile.write(
    "mixed2.wav",
    fs1,
    Y[:, 1].astype(np.int16)
)

print("\nSaved mixed1.wav and mixed2.wav")


# Target SNR
SNR = 5

'''
good reconstruction - SNR = 5: 
both reconstructions are clearly the 
original track but with some N heavy

Bad reconstruction SNR <= 1
'''


signal_power = np.linalg.norm(Y)**2
noise_power = signal_power / SNR
num_elements = Y.size

# Each element's std
sigma = np.sqrt(noise_power / num_elements)

# Generate noise with same dimensions as Y
N = np.random.normal(
    loc=0,
    scale=sigma,
    size=Y.shape
)

# Noise
Z = Y + N


# Calculate real SNR
actual_SNR = (
    np.linalg.norm(Y)**2 /
    np.linalg.norm(N)**2
)

print("\nNoise:")
print("Target SNR:", SNR)
print("Actual SNR:", actual_SNR)
print("Noise dimensions:", N.shape)


# Save noisy audio tracks
wavfile.write(
    "noisy1.wav",
    fs1,
    Z[:, 0].astype(np.int16)
)

wavfile.write(
    "noisy2.wav",
    fs1,
    Z[:, 1].astype(np.int16)
)

print("\nSaved noisy1.wav and noisy2.wav")

# first track
track = 0

figure, ax = plt.subplots(3, 1, figsize=(12, 8))

# Mixed signal
ax[0].plot(Y[:, track])
ax[0].set_title("Mixed audio")
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Amplitude")

# Noise
ax[1].plot(N[:, track])
ax[1].set_title("Noise")
ax[1].set_xlabel("Sample")
ax[1].set_ylabel("Amplitude")

# Noisy mixed signal
ax[2].plot(Z[:, track])
ax[2].set_title("Noisy mixed audio")
ax[2].set_xlabel("Sample")
ax[2].set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig("noise_comparison.png")
plt.close()

print("\nSaved noise_comparison.png")

# Inverse mixing matrix
W_inv = np.linalg.inv(W)

# Demix noisy and mixed signal
X_reconstructed = Z @ W_inv

print("\nReconstructed signal dimensions:")
print(X_reconstructed.shape)


# Save the reconstructed audio tracks
wavfile.write(
    "reconstructed1.wav",
    fs1,
    X_reconstructed[:, 0].astype(np.int16)
)

wavfile.write(
    "reconstructed2.wav",
    fs1,
    X_reconstructed[:, 1].astype(np.int16)
)

print("Saved reconstructed1.wav and reconstructed2.wav")


fft_rec1 = fft(X_reconstructed[:, 0])
fft_rec2 = fft(X_reconstructed[:, 1])

freq_rec1 = fftfreq(
    n=len(X_reconstructed[:, 0]),
    d=1/fs1
)

freq_rec2 = fftfreq(
    n=len(X_reconstructed[:, 1]),
    d=1/fs2
)

figure, ax = plt.subplots(2, 2, figsize=(12, 8))

# Reconstructed wave forms
ax[0, 0].plot(X_reconstructed[:, 0])
ax[0, 0].set_title("Reconstructed StarWars")
ax[0, 0].set_xlabel("Sample")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 1].plot(X_reconstructed[:, 1])
ax[0, 1].set_title("Reconstructed CantinaBand")
ax[0, 1].set_xlabel("Sample")
ax[0, 1].set_ylabel("Amplitude")

# Fourier transform of reconstructoin
ax[1, 0].plot(freq_rec1, np.abs(fft_rec1))
ax[1, 0].set_title("Reconstructed StarWars Fourier transform")
ax[1, 0].set_xlabel("Frequency (Hz)")
ax[1, 0].set_ylabel("Magnitude")
ax[1, 1].plot(freq_rec2, np.abs(fft_rec2))
ax[1, 1].set_title("Reconstructed CantinaBand Fourier transform")
ax[1, 1].set_xlabel("Frequency (Hz)")
ax[1, 1].set_ylabel("Magnitude")

plt.tight_layout()
plt.savefig("reconstructed_audio.png")
plt.close()

print("Saved reconstructed_audio.png")


# 3.3.2

theta = np.pi / 3

# ALphas
alphas = np.linspace(0.05, 0.95, 100)

# SNRs
SNR_values = [30, 10, 5, 1]

# Reproducible noice
np.random.seed(1)

plt.figure(figsize=(10, 6))

for SNR in SNR_values:

    errors = []

    for alpha in alphas:

        # Rotation matrix
        A = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        #Scaling matrix
        B = np.array([
            [alpha, 0],
            [0, 1 - alpha]
        ])

        # Mixing matrix
        W = A @ B

        # Mix original signals
        Y = X @ W

        # Calculate required N power as before
        signal_power = np.linalg.norm(Y)**2
        noise_power = signal_power / SNR

        # Std of N as before
        num_elements = Y.size
        sigma = np.sqrt(noise_power / num_elements)

        # Create gaussian N
        N = np.random.normal(
            loc=0,
            scale=sigma,
            size=Y.shape
        )

        # Add noise
        Z = Y + N

        # Demix -- whit noise
        W_inv = np.linalg.inv(W)
        X_reconstructed = Z @ W_inv

        # Reconstruction error from "section 2"
        error = (
            np.linalg.norm(X_reconstructed - X)
            / np.linalg.norm(X)
        )

        errors.append(error)

    # Plot error as a function of alpha
    plt.plot(
        alphas,
        np.log(errors),
        label=f"SNR = {SNR}"
    )

plt.xlabel("Alpha")
plt.ylabel("Reconstruction error")
plt.title("Reconstruction error as a function of alpha")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig("reconstruction_error.png")
plt.close()

print("Saved reconstruction_error.png")







