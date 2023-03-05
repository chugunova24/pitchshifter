import numpy as np
from scipy.interpolate import interp1d
from frames import Frames

"""
    Is the implementation of the Pitch Shifter algorithm.
    
    :param
        inputVector: sound wave, specified as a vector.
        windowSize: the size of one frame.
        hop: hop distance/size
        step: the step size. 
    
    :return
        resampled and interpolated output.

"""

def pitchShift(inputVector, windowSize, hop, step):
    pi = np.pi
    exp = np.exp
    alpha = np.power(2, (step / 12))
    hopOut = round(alpha * hop)

    # numpy array of accumulated phases.
    phaseCumulative = 0
    # numpy array of all of the previous frames phase information.
    previousPhase = 0

    # Hanning window
    wn = np.hanning(windowSize * 2 + 1)
    wn = wn[2::2]


    # Initialization
    frame = Frames()
    # division of the vector into frames
    y, numberFramesInput = frame.create_frames(x=inputVector, hop=hop, window_size=windowSize)
    numberFramesOutput = numberFramesInput
    outputy = np.zeros((numberFramesOutput, windowSize))

    for index in range(0, numberFramesInput):

        # Analysis.
        currentFrame = y[index]
        currentFrameWindowed = currentFrame * wn / np.sqrt(((windowSize/hop)/2))
        # The frame made of N samples is then transformed with a Fast Fourier Transform (FFT)
        currentFrameWindowedFFT = np.fft.fft(currentFrameWindowed)
        magFrame = np.abs(currentFrameWindowedFFT)
        phaseFrame = np.angle(currentFrameWindowedFFT)

        # Processing
        deltaPhi = phaseFrame - previousPhase
        previousPhase = phaseFrame
        # Bin frequency
        omegaBin = np.arange(windowSize)/windowSize
        # The frequency deviation
        deltaPhiPrime = deltaPhi - hop * omegaBin
        # The wrapped frequency deviation
        deltaPhiPrimeMod = np.mod(deltaPhiPrime + pi, 2 * pi) - pi
        # The true frequency
        trueFreq = np.array(omegaBin + deltaPhiPrimeMod/hop, dtype=np.int16)
        phaseCumulative = phaseCumulative + hopOut * trueFreq  # numpy array of accumulated phases.

        # Synthesis
        outputMag = magFrame
        # The inverse discrete Fourier transform (IDFT) is performed on each frame spectrum.
        outputFrame = np.real(np.fft.ifft(outputMag * np.exp(1j * phaseCumulative)))
        outputy[index] = outputFrame * wn / np.sqrt(((windowSize/hopOut)/2))


    # Resampling
    outputTimeStretched = frame.fusion_frames(frames_matrix=outputy, hop=hopOut)
    shapeVector = outputTimeStretched.shape[0]
    interpolator = interp1d(x=np.arange(0, shapeVector), y=outputTimeStretched, kind='linear')
    resample = np.linspace(start=0, stop=shapeVector - 1, num=int(shapeVector))

    return interpolator(resample)