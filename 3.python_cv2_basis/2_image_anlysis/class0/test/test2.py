import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin,lfilter
import audioread
import librosa
import soundfile as sf

audio_data, sample_rate = librosa.load('class0\\test\\test.wav', sr=None)

print("")
print("=== 音频信息 ===")
print(f"音频地址：class0\\test\\test.wav")
print(f"采样率: {sample_rate} Hz")
print(f"时长: {len(audio_data)/sample_rate:.2f} 秒")
print(f"样本数: {len(audio_data)}")
print(f"数据类型: {audio_data.dtype}")
print(f"数值范围: [{audio_data.min():.3f}, {audio_data.max():.3f}]")

N = len(audio_data)
x = []

for i in audio_data:
    x.append(10*i)

time = np.arange( 0 , N , 1)

# plt.plot(time , x , 'r-' , alpha = 0.8)
# plt.show()

x = audio_data

P = firwin(32 , 0.1)
S = firwin(16 , 0.1)
Sh = firwin(16 , 0.1)
Xp = np.zeros(32)
W = np.zeros(24)
Xw = np.zeros(24)
Xs = np.zeros(16)
Y = np.zeros(16)
Xf = np.zeros(24)

d = np.zeros(N)
y = np.zeros(N)
ys = np.zeros(N)
e = np.zeros(N)
xf = np.zeros(N)

u = 0.001

for k in range(N):
    Xp[1:] = Xp[:-1]
    Xp[0] = x[k]
    d[k] = np.dot(Xp, P)

    Xw[1:] = Xw[:-1]
    Xw[0] = x[k]
    y[k] = np.dot(Xw, W)

    Y[1:] = Y[:-1]
    Y[0] = y[k]
    ys[k] = np.dot(Y, S)

    e[k] = d[k] - ys[k]

    Xs[1:] = Xs[:-1]
    Xs[0] = x[k]
    xf[k] = np.dot(Xs, Sh)

    Xf[1:] = Xf[:-1]
    Xf[0] = xf[k]
    W = W + u * e[k] * Xf

t = time

plt.subplot(4, 1, 1)
plt.plot(t, d, 'r-', alpha=0.8, label='input signal')
plt.legend()
plt.xlabel('Time')
plt.ylabel('dB')

plt.subplot(4, 1, 2)
plt.plot(t, d, 'r-', alpha=0.8, label='Expected signal')
plt.plot(t, ys, 'b-', alpha=0.8, label='Anti noise')
plt.legend()
plt.xlabel('Time')
plt.ylabel('dB')

plt.subplot(4, 1, 3)
plt.plot(t, ys, 'b-', alpha=0.8, label='ys')
plt.legend()
plt.xlabel('Time')
plt.ylabel('dB')

plt.subplot(4, 1, 4)
plt.plot(t, e, 'g-', alpha=0.8, label='Error signal')
plt.legend()
plt.xlabel('Time')
plt.ylabel('dB')
plt.show()


e_normalized = e  # 可选：归一化到 [-1, 1]
sf.write('class0/test/output_e.wav', e_normalized, sample_rate)
