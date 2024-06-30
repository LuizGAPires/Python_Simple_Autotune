from scipy import signal
import numpy as np

def f_1(x):
    f_0 = 1
    return np.sin(x * np.pi * 2 * f_0)

def f_2(x):
    f_0 = 1
    envelope = lambda x: np.exp(-x)
    return np.sin(x * np.pi * 2 * f_0) * envelope(x)

def ACF(f, W, t, lag):
    return np.sum(
        f[t : t + W] *
        f[lag + t : lag + t + W]
    )

def DF(f, W, t, lag):
    return ACF(f, W, t, 0) + ACF(f, W, t + lag, 0) - (2 * ACF(f, W, t, lag))

def CMNDF(f, W, t, lag):
    if lag == 0:
        return 1
    return DF(f, W, t, lag) / np.sum([DF(f, W, t, j + 1) for j in range(lag)]) * lag

def detect_pitch(f, W, t, sample_rate, bounds, thresh=0.1):
    #ACF_vals = [ACF(f, W, t, i) for i in range(*bounds)]
    #sample = np.argmax(ACF_vals) + bounds[0]
    #DF_vals = [DF(f, W, t, i) for i in range(*bounds)]
    #sample = np.argmin(DF_vals) + bounds[0]
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(*bounds)]
    sample = None
    for i, val in enumerate(CMNDF_vals):
        if val < thresh:
            sample = i + bounds[0]
            break
    if sample is None:
        sample = np.argmin(CMNDF_vals) + bounds[0]
    return sample_rate / sample

def main():
    sample_rate = 500
    start = 0
    end = 5
    num_samples = int(sample_rate * (end - start) + 1)
    window_size =  200
    bounds = [20, num_samples // 2]

    x =  np.linspace(start, end, num_samples)
    print(detect_pitch(f_1(x), window_size, 1, sample_rate, bounds))
    print(detect_pitch(f_2(x), window_size, 1, sample_rate, bounds))

main()