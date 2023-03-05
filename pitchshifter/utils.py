import collections
import numpy as np

def scalar_len(a):
    if isinstance(a, collections.abc.Iterable):
        return len(a)
    else:
        return 1

def stereo_to_mono(inputVector):
    """
        Takes in an 2d array of stereo samples and returns a mono numpy
        array of dtype np.int16.
    """
    LEFT = 0
    RIGHT = 1
    channels = scalar_len(inputVector[0])
    if channels == 1:
        mono_samples = np.asarray(inputVector,
                                  dtype=np.int16)

    elif channels == 2:
        mono_samples = np.asarray(
            [(sample[RIGHT] + sample[LEFT]) / 2 for sample in inputVector],
            dtype=np.int16
        )

    else:
        raise Exception('Must be mono or stereo')

    return mono_samples


def compute_parameters(time_stretch_ratio):
    """
    :param time_stretch_ratio
    :return computed parameters hop and step
    """

    if 0 < time_stretch_ratio < 1:
        return int(512 - time_stretch_ratio*100), int(time_stretch_ratio * 10 - 10)

    elif time_stretch_ratio >= 1:
        return int(512 * (np.absolute(time_stretch_ratio-9) / 10)+99), time_stretch_ratio

    else:
        raise Exception('The algorithm parameter "r" must be in the ranges (0 < r < 1) or (1 <= r)')
