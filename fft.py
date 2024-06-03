import numpy as np
from scipy.fft import fft, fftfreq

def FastFourierTransformNew(Depth, Proxy, pad=5):

    N = len(Proxy)
    NFFT = int(pad*N)
    Fs = round(np.median(np.diff(Depth)), 2)

    yf = fft(Proxy, n=NFFT)
    f = fftfreq(NFFT, Fs)
    freqFFT = f[:NFFT // 2]

    halfFFT = yf[0:NFFT//2]
    ampFFT_PSD = 1/N * np.abs(halfFFT)**2

    return freqFFT, ampFFT_PSD