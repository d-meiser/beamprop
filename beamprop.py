import numpy as np
import matplotlib.pyplot as plt


class Field():
    def __init__(self, positions, amplitudes):
        self.positions = np.copy(positions)
        self.amplitudes = np.copy(amplitudes)


def _unpack_field_data(packed_data):
    len = int(np.sqrt(packed_data.shape[0] / 4))
    assert(len * len * 4 == packed_data.shape[0])
    new_shape = [len, len]
    x = np.reshape(packed_data[0::4], new_shape)
    y = np.reshape(packed_data[1::4], new_shape)
    positions = np.array([x, y])
    amplitudes = packed_data[2::4] + 1.0j * packed_data[3::4]
    amplitudes = np.reshape(amplitudes, new_shape)
    return Field(positions, amplitudes)


def get_field_data(filename):
    packed_data = np.fromfile(filename, dtype=np.float64, sep=' ')
    return _unpack_field_data(packed_data)


initial_state = get_field_data('initial_state.dat')
n = 1
propagated_state = get_field_data('field_' + str(n) + '.dat')

plt.plot(initial_state.positions[0, :, 64],
         np.abs(initial_state.amplitudes[:, 64])**2)
plt.plot(propagated_state.positions[0, :, 64],
         np.abs(propagated_state.amplitudes[:, 64])**2)


