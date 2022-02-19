import numpy as np
import os
import ddsp


def unit_midi_to_hz(x):
    x = 127.0 * x - 69.0
    x = x / 12.0
    x = 440.0 * 2 ** x
    return x


STEPS = 1500
THIRD = int(STEPS / 3)
FOURTH = int(STEPS / 4)
noiseamp = 0.05
x = 0.54330707 * np.ones(FOURTH)
x = np.concatenate((x, 0.5669289 * np.ones(FOURTH)))
x = np.concatenate((x, 0.4488189 * np.ones(FOURTH)))
x = np.concatenate((x, 0.5784174 * np.ones(FOURTH)))
ld = 0.5 * np.ones(x.shape)
#x = unit_midi_to_hz(x)
x = np.reshape(x, (1, x.shape[0]))
ld = np.reshape(ld, (1, ld.shape[0]))
x = np.concatenate((x, ld))
x[0] = x[0] + noiseamp * np.random.randn(x.shape[1])
#x[1] = x[1] + noiseamp * np.random.randn(x.shape[1])
np.save(".\\signal.npy", x)

cmd = 'python .\\face_decoder.py violin .\\signal.npy .\\callerout-' + str(STEPS) + '.wav'
os.system(cmd)
