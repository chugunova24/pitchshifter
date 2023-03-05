import numpy as np


class Frames:

    def create_frames(self, x, hop, window_size):
        """
        :param x: input vector of sound wave
        :param hop: size of hop
        :param window_size: size of one frame
        :return: division of the vector into frames
        """

        number_slices = int(np.floor((len(x) - window_size) / hop))
        x = x[1:(number_slices * hop + window_size)]
        vector_frames = np.zeros((int(np.floor(len(x) / hop)), window_size))

        for index in range(0, number_slices):
            index_timeStart = (index) * hop
            index_timeEnd = (index) * hop + window_size
            vector_frames[index] = x[index_timeStart: index_timeEnd]

        return vector_frames, number_slices

    def fusion_frames(self, frames_matrix, hop):
        """
        :param frames_matrix: input matrix of frames
        :param hop: size of hop
        :return: 1d-array
        """

        size_matrix = np.shape(frames_matrix)
        number_frames = size_matrix[0]
        size_frames = size_matrix[1]
        vector_time = np.zeros((number_frames * hop - hop + size_frames))
        time_index = 0

        for index in range(0, number_frames):
            vector_time[time_index:(time_index + size_frames)] += frames_matrix[index]
            time_index += hop

        return vector_time









