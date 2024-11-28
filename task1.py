import numpy as np
import matplotlib.pyplot as plt

# Constants
n = 50
samples = np.arange(n)
sampling_rate = 100
wave_velocity = 8000

# Function to generate Signal A and Signal B
def generate_signals(frequency=5):
    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz
    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40]
    amplitudes2 = [0.3, 0.2, 0.1]

    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_signal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                             for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_signal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                             for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))

    signal_A = original_signal + noise_for_signal_A
    noisy_signal_B = signal_A + noise_for_signal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)

    return signal_A, signal_B

# Generate signals
signal_A, signal_B = generate_signals()

# Plot Signal A
plt.figure(figsize=(10, 5))
plt.stem(samples, signal_A, linefmt="blue", markerfmt="bo", basefmt=" ")
plt.title("Signal A (Station A)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()



# DFT implementation
def DFT(signal):
    N = len(signal)
    real_part = np.zeros(N)
    imag_part = np.zeros(N)
    for i in range(N):
        for j in range(N):
            real_part[i] += signal[j] * np.cos(2 * np.pi * i * j / N)
            imag_part[i] += -signal[j] * np.sin(2 * np.pi * i * j / N)
    return real_part, imag_part

# Compute DFT for Signal A
fA_real, fA_imag = DFT(signal_A)
fA_magnitude = np.sqrt(fA_real**2 + fA_imag**2)

plt.figure(figsize=(10, 5))
plt.stem(samples, fA_magnitude, linefmt="blue", markerfmt="bo", basefmt=" ")
plt.title("Frequency Spectrum of Signal A")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Compute DFT for Signal B
fB_real, fB_imag = DFT(signal_B)
fB_magnitude = np.sqrt(fB_real**2 + fB_imag**2)

# Plot Signal B
plt.figure(figsize=(10, 5))
plt.stem(samples, signal_B, linefmt="red", markerfmt="ro", basefmt=" ")
plt.title("Signal B (Station B)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.stem(samples, fB_magnitude, linefmt="red", markerfmt="ro", basefmt=" ")
plt.title("Frequency Spectrum of Signal B")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Compute Cross-Correlation in Frequency Domain
cross_real = fA_real * fB_real + fA_imag * fB_imag
cross_imag = -fA_real * fB_imag + fA_imag * fB_real
signal_cross = [cross_real, cross_imag]

# IDFT implementation
def IDFT(signal):
    real_part = signal[0]
    imag_part = signal[1]
    N = len(real_part)
    time_domain_signal = np.zeros(N)
    for i in range(N):
        for j in range(N):
            time_domain_signal[i] += (real_part[j] * np.cos(2 * np.pi * i * j / N)
                                      - imag_part[j] * np.sin(2 * np.pi * i * j / N))
    return time_domain_signal / N

# Compute Cross-Correlation in Time Domain
CrossCorrInTimeDomain = IDFT(signal_cross)
shifted_index = np.arange(-n//2, n//2)

plt.figure(figsize=(10, 5))
plt.stem(shifted_index, CrossCorrInTimeDomain, linefmt="green", markerfmt="go", basefmt=" ")
plt.title("DFT-Based Cross-Correlation")
plt.xlabel("Sample Lag")
plt.ylabel("Correlation Magnitude")
plt.grid()
plt.show()

# Calculate Sample Lag and Distance
sample_lag = np.argmax(CrossCorrInTimeDomain)
sample_lag = n - sample_lag if sample_lag > n // 2 else -sample_lag
print("Found after DFT-based Cross-Correlation: ", sample_lag)

distance = sample_lag * wave_velocity * (1 / sampling_rate)
print("Distance: ", distance)
