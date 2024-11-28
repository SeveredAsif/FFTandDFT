import numpy as np
import matplotlib.pyplot as plt
import time 

n = [2**2,2**3,2**4,2**5,2**6,2**7]
n
sequences = []
for length in n:
    sequences.append(np.random.randint(0,100,length))
print(sequences[5])
def DFT(signal):
    N = len(signal)
    W_N = np.array(np.zeros(shape=(N,N)), dtype=complex)
    w_n = np.exp(-1*2j*np.pi/N)

    #matrix build
    for i in range(N):
        for j in range(N):
            W_N[i][j] = w_n ** (i*j)
    
    #print(W_N)
    DFTmatrix = np.matmul(W_N,signal)
    return DFTmatrix
def IDFT(signal):
    N = len(signal)
    W_N = np.array(np.zeros(shape=(N,N)), dtype=complex)
    w_n = np.exp(2j*np.pi/N)

    #matrix build
    for i in range(N):
        for j in range(N):
            W_N[i][j] = w_n ** (i*j)
    W_N = W_N/N
    IDFTmatrix = np.matmul(W_N,signal)
    return IDFTmatrix
def FFT(signal):
    N = len(signal)
    if (N==1):
        return signal
    even = signal[::2]
    odd = signal[1::2]
    even_fft = FFT(even)
    odd_fft = FFT(odd)
    return_arr = np.array(np.zeros(N), dtype=complex)
    r = N//2
    for k in range(r):
        twiddler_factor = np.exp(-2j * np.pi * k/N)
        return_arr[k] = even_fft[k] + twiddler_factor * odd_fft[k]
        return_arr[k+r] = even_fft[k] - twiddler_factor * odd_fft[k]
    return return_arr


def IFFT(signal):
    N = len(signal)
    if (N==1):
        return signal
    even = signal[::2]
    odd = signal[1::2]
    even_fft = FFT(even)
    odd_fft = FFT(odd)
    return_arr = np.array(np.zeros(N), dtype=complex)
    r = N//2
    for k in range(r):
        twiddler_factor = np.exp(2j * np.pi * k/N)
        return_arr[k] = even_fft[k] + twiddler_factor * odd_fft[k]
        return_arr[k+r] = even_fft[k] - twiddler_factor * odd_fft[k]
    return return_arr/N

# Timing function
def measure_runtime(algorithm, signal, num_runs=10):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        algorithm(signal)
        total_time += time.time() - start_time
    return total_time / num_runs

# Store average runtimes
dft_times = []
fft_times = []
idft_times = []
ifft_times = []

for seq in sequences:
    dft_times.append(measure_runtime(DFT, seq))
    fft_times.append(measure_runtime(FFT, seq))
    idft_times.append(measure_runtime(IDFT, seq))
    ifft_times.append(measure_runtime(IFFT, seq))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(n, dft_times, label="DFT", marker='o')
plt.plot(n, fft_times, label="FFT", marker='o')
plt.plot(n, idft_times, label="IDFT", marker='o')
plt.plot(n, ifft_times, label="IFFT", marker='o')

plt.xlabel("Input Size (n)")
plt.ylabel("Average Runtime (seconds)")
plt.title("Runtime Comparison of DFT/FFT and IDFT/IFFT")
plt.legend()
plt.grid()
plt.show()