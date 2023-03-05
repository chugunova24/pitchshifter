import argparse
import sys
import numpy as np
import scipy
from pitchshift import pitchShift
from utils import stereo_to_mono, compute_parameters


"""
    Description:
        This project is the implementation of the Guitar Pitch Shifter algorithm.
    
    Link to the original article:
        https://www.guitarpitchshifter.com/algorithm.html
    
"""


def main(args={}):
    PATH = args.source
    windowSize = 1024
    hop, step = compute_parameters(float(args.time_stretch_ratio))

    # Try to open the wav file and read it
    try:
        frameFreq, inputVector = scipy.io.wavfile.read(PATH)
    except:
        print(f'File {args.source} does not exist')
        sys.exit(-1)

    stereo_to_mono(inputVector)
    result = pitchShift(inputVector=inputVector, windowSize=windowSize,
                        hop=hop, step=step)
    scipy.io.wavfile.write(args.output, frameFreq, np.asarray(result, dtype=np.int16))



def cli():
    parser = argparse.ArgumentParser(
        description="The program is designed to compress and stretch audio twice.")
    parser.add_argument('source', metavar='S', help='source .wav file')
    parser.add_argument('output', metavar='O', help='output .wav file')
    parser.add_argument('time_stretch_ratio', metavar='TSR', help='algorithm parameter')
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli()