import sys

sys.path.insert(1,
                '/home/vesathya/ModulationClassification/Aug2020/Code/JournalPaperSNRPartitioning/CFOSweep/DatasetGeneration_GnuRadio/AllertonDataGeneration/')

from transmitters import transmitters
from source_alphabet_New import source_alphabet
# import analyze_stats
from gnuradio import channels, gr, blocks, analog
import numpy as np
# import numpy.fft, gzip
import random
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import time
from itertools import product
import pickle
import datetime


# clockeffects_dict{'CFOMaxdev'}=500
# # Say signal CF is 900MHZ - LO freq is 900MHZ. The CFOMax dev at 900MHz is 500Hz.
# #The ADC clocks rates from datasheet of N310 is 122.88MHz.
# # SRO Max dev is therefore calculated as 500*122.88/900 = 68.27 = 70Hz approx.
# clockeffects_dict{'SROMaxdev'}=70
# clockeffects_dict{'SROstddev'}=0.01
# clockeffects_dict{'CFOstddev'}=0.01

def clockEffects(samples, clockeffects_dict, samp_rate, seed):
    np.random.seed(seed)  # setting the random seed
    # To check if the seed has been set, use the command np.random.get_state()[1][0]
    Ts = 1 / samp_rate  # sample time
    samples_length = samples.shape[0]
    energy = np.sum((np.abs(samples)))
    samples = samples / energy
    ferr_bias_CFO = np.random.uniform(-clockeffects_dict['CFOMaxdev'], clockeffects_dict['CFOMaxdev'])
    clockerror_randomsamples = np.random.randn(samples_length, )
    CFO_cum = np.cumsum(clockerror_randomsamples * clockeffects_dict['CFOstddev']) + ferr_bias_CFO
    n = np.arange(0, samples_length)
    CFO_mult_n = np.multiply(CFO_cum, n)

    # cosine_arr = np.cos(2 * np.pi * CFO_mult_n * Ts)
    # sine_arr = np.sin(2 * np.pi * CFO_mult_n * Ts)
    # realval_withCFO = np.multiply(np.real(samples), cosine_arr) - \
    #                  np.multiply(np.imag(samples), sine_arr)
    # imagval_withCFO = np.multiply(np.real(samples), sine_arr) + \
    #                  np.multiply(np.imag(samples), cosine_arr)
    # potentially replace lines 38 to 41 with the following two lines
    exp_val = np.exp(1j * 2 * np.pi * CFO_mult_n * Ts)
    samples_withCFO = np.multiply(exp_val, samples)
    realval_withCFO = np.real(samples_withCFO)
    imagval_withCFO = np.imag(samples_withCFO)
    # Apply SRO
    # Ferr is difference between the TX DAC and RX ADC clocks rates.
    # we are resampling for this difference.
    SROCFObias_scaling = clockeffects_dict['SROMaxdev'] / clockeffects_dict['CFOMaxdev']
    # Note that the source of error is common for both CFO and SRO - XO crystal.
    ferr_bias_SRO = ferr_bias_CFO * SROCFObias_scaling
    # np.random.uniform(-clockeffects_dict['SROMaxdev'], clockeffects_dict['SROMaxdev'])
    SRO_cum = np.cumsum(clockerror_randomsamples * clockeffects_dict['SROstddev']) + ferr_bias_SRO
    # n = np.arange(0,n = np.arange(0,L))
    # CFO_SRO_mult_n = np.multiply(SRO_cum, n)
    Ts_range = np.linspace(0, samples_length * Ts, samples_length)  # np.arange(0, samples_length * Ts, Ts)
    Fs_prime = SRO_cum  # Diff freq error at different time steps.
    Ts_prime = np.reciprocal(Fs_prime + samp_rate)
    Ts_prime_range = np.cumsum(Ts_prime)
    f_imag = interpolate.interp1d(Ts_range, imagval_withCFO, fill_value='extrapolate')
    f_real = interpolate.interp1d(Ts_range, realval_withCFO, fill_value='extrapolate')
    samples_clockeffect_real = f_real(Ts_prime_range)
    samples_clockeffect_imag = f_imag(Ts_prime_range)
    samples_clockeffect = samples_clockeffect_real + 1j * samples_clockeffect_imag
    return samples_clockeffect


def phaseOffset(samples, seed):
    # theta = np.random.uniform(0,2*np.pi)
    np.random.seed(seed)  # setting the random seed
    # To check if the seed has been set, use the command np.random.get_state()[1][0]
    phaseOffsetVal = np.random.uniform(0, 2 * np.pi)
    samples_phaseOffset = np.multiply(samples, np.exp(1j * phaseOffsetVal))
    return samples_phaseOffset


dataset = {}
if len(sys.argv) < 2:
    numFrames_permodsnr = 1000  # Number of frames per modulation type and SNR.
else:
    numFrames_permodsnr = int(sys.argv[1])
frame_length = 128  # length of a frame.
snr_levels = range(-20, 21, 2)
transients = 1000  # typically after channel, clock effects are applied, the output has trasients for
# the first 500 odd samples that need to be ignored.
# In a long array fo samples post effects such as clock,AWGN,fading etc., we iteratively carve chunks of samples equal to frame_length.
# and jump by an increment that is decided by the samples_incrementscale.
samples_incrementscale = 0.05

audio_rate = 44.1e3
# The audio source file has almost no signal for the first few seconds.
analog_transients = int(audio_rate * 5)
# Samples of digitalsample_len will be passed through modulation, AWGN, ClockEffects, Fading blocks.
# A single frame will be carved out with a randomly placed window,
# with a minimum offset of transients out of signal of length digitalsample_len.
digitalsample_len = frame_length * 100 + transients

# clean - only the modulated signal
# AWGN - additive thermal noise applied to clean modulated signal for specified SNR levels.
# AWGN_clock - AWGN and errors due to clock drifts - center frequency offset (CFO) and sample rate offset (SRO)
# applied.
datasettype_list = ["clean", "AWGN", "AWGN_clock", "all"]
fD = 70  # ETU70, max doppler ferquency/frequency devaition - 70Hz.
# Tsym for LTE is 71.3 microseconds.
# Samp_rate caluclated as = 8/71.3 (8 is the upsampling factor - i.e. umber of samples per symbol)
samp_rate = 112.2e3
Ts = 1 / samp_rate  # sample time
# delays in nanoseconds converted into fractional sample delays. Ech sample delay Ts = 9 micro seconds approx.
delays = [val_temp * (1e-9) * samp_rate for val_temp in [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]]
mags_dB = [-1, -1, -1, 0, 0, 0, -3, -5, -7]
mags = [10 ** (mags_val / 20.0) for mags_val in mags_dB]
ntaps = 8
numSinusoids = 8
Kfactor = 4
# noise_amp = 10 ** (-snr / 20.0)
clockeffects_dict = {}
clockeffects_dict['CFOMaxdev'] = 500
# Say signal CF is 900MHZ - LO freq is 900MHZ. The CFOMax dev at 900MHz is 500Hz.
# The ADC clocks rates from datasheet of N310 is 122.88MHz.
# SRO Max dev is therefore calculated as 500*122.88/900 = 68.27 = 70Hz approx.
clockeffects_dict['SROMaxdev'] = 70
clockeffects_dict['SROstddev'] = 0.01
clockeffects_dict['CFOstddev'] = 0.01
LOS = False  # Rayleigh channel, no LOS component

# Analog data is large and takes time to read. Therefore done once in the beginning and random windows
# used to extract samples for each input frame.
# NOTE: If there are issues loading the large audio file such as RAM being filled up,
# you can use any smaller sized audio file.


tb = gr.top_block()
src_cont = source_alphabet("continuous", 1000)
snk_cont = blocks.vector_sink_f()
tb.connect(src_cont, snk_cont)
tb.run()
# data = np.array(snk.data(), dtype=np.complex64)
# print(data[0:10])


# tb_analogdata = gr.top_block()
# src_analogdata = source_alphabet("continuous", 1000)
# snk_analogdata = blocks.vector_sink_f()
# tb_analogdata.connect(src_analogdata, snk_analogdata)
# tb_analogdata.run()
analogdata = np.array(snk_cont.data(), dtype=np.complex64)
len_analogdata = analogdata.shape[0]
# tb = gr.top_block()
# src = source_alphabet("continuous", 1000)
# snk = blocks.vector_sink_f()
# tb.connect(src,snk)
# tb.run()
# data = np.array(snk.data(), dtype=np.complex64)


# analogdata = np.array(snk.data(), dtype=np.complex64)
# len_analogdata = analogdata.shape[0]


seed = 0
# datasettype = datasettype_list[3]
print("changed datasettype_list to all")
for datasettype in datasettype_list:#datasettype_list:
    print("Datasettype is :", datasettype)
    for modulation_flavour in list(transmitters.keys()):
        for modulation_type, snr in product(list(transmitters[modulation_flavour]), snr_levels):
            dataset[(modulation_type.modname, snr)] = np.zeros([numFrames_permodsnr, 2, frame_length], dtype=np.float32)

    for modulation_flavour in list(transmitters.keys()):

        for modulation_type in list(transmitters[modulation_flavour]):

            for snr in snr_levels:
                ct = datetime.datetime.now()
                print("Generating data for Modulation type: ", modulation_type.modname," and snr: ", snr, " at time ", ct)
                frame_idx = -1
                # seed = seed + 1  # different seed for different frames
                noise_amp = 10 ** (-snr / 20.0)  # you need to divide by 20 instead of 10 to get the correct SNR.
                while frame_idx < numFrames_permodsnr:
                    # print(modulation_flavour,modulation_type,snr,frame_idx)

                    # Random seed generated - to seed digital source generation, analog source generation,
                    # AWGN noise generation, fading channel simulation, CFO/SRO simulation and phase offset simulation.
                    # seed = seed_itr
                    np.random.seed(seed)  # setting the random seed
                    # To check if the seed has been set, use the command np.random.get_state()[1][0]
                    mod = modulation_type()  # in the modified transmitter file, change
                    add_block = blocks.add_vcc(1)
                    noise_block = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amp, seed)

                    snk = blocks.vector_sink_c()
                    fading_block = channels.selective_fading_model(numSinusoids, fD / samp_rate, LOS, Kfactor, \
                                                                   seed, delays, mags, ntaps)
                    tb = gr.top_block()

                    if modulation_flavour == "discrete":
                        np.random.seed(seed)
                        src = source_alphabet("discrete", digitalsample_len, seed)
                    elif modulation_flavour == "continuous":
                        # the first 5 seconds and last 60 seconds of the audio recording is ignored. The idx is the staring
                        # point for carving a sample stream for continuous type modulation.
                        np.random.seed(seed)
                        idx_analogdata = np.random.randint(analog_transients, len_analogdata - int(audio_rate * 60))
                        # For analog source, we simply are using the real part of the audio signal.
                        # This is also consistent with the implementation by Oshea who also uses only the real part,when using
                        # a complextofloat block.
                        analogdata_sample = np.real(analogdata[idx_analogdata:idx_analogdata + digitalsample_len])
                        src = blocks.vector_source_f(analogdata_sample, False, 1, [])
                    if datasettype == 'clean':
                        tb.connect(src, mod, snk)
                    elif (datasettype == 'AWGN') or (datasettype == 'AWGN_clock'):
                        # Apply AWGN only now for both the datasettype
                        tb.connect(src, mod)
                        tb.connect(noise_block, (add_block, 1))
                        tb.connect(mod, (add_block, 0))
                        tb.connect(add_block, snk)
                    elif datasettype == 'all':
                        tb.connect(noise_block, (add_block, 1))
                        tb.connect(src, mod, (add_block, 0))
                        tb.connect(add_block, fading_block, snk)
                    tb.run()
                    # tb.stop()
                    samples_allEffects = np.array(snk.data(), dtype=np.complex64)
                    # remove transients from the samples post fading.
                    samples_allEffects = samples_allEffects[transients:]
                    len_sampAlleffects = samples_allEffects.shape[0]
                    np.random.seed(seed)
                    incr_idx = np.random.randint(0,int(len_sampAlleffects*samples_incrementscale))
                    start_idx = 0
                    start_idx = start_idx + incr_idx
                    frame_idx = frame_idx + 1
                    while (start_idx +frame_length < len_sampAlleffects) and (frame_idx < numFrames_permodsnr):
                        frame_allEffects = samples_allEffects[start_idx:start_idx+frame_length]

                        if (datasettype == 'AWGN_clock') or (datasettype == 'all'):
                            frame_allEffects = clockEffects(frame_allEffects, clockeffects_dict, samp_rate, seed)
                        if (datasettype == 'AWGN') or (datasettype == 'AWGN_clock') or (datasettype == 'all'):
                            frame_allEffects = phaseOffset(frame_allEffects, seed)

                        dataset[(modulation_type.modname, snr)][frame_idx, 0, :] = np.real(frame_allEffects)
                        dataset[(modulation_type.modname, snr)][frame_idx, 1, :] = np.imag(frame_allEffects)

                        seed = seed + 1
                        np.random.seed(seed)
                        incr_idx = np.random.randint(0,int(len_sampAlleffects*samples_incrementscale))
                        start_idx = start_idx + incr_idx + frame_length
                        #print(start_idx,frame_idx)
                        frame_idx = frame_idx + 1

                frame_idx = frame_idx - 1
                seed = seed - 1



                    # if (datasettype == 'AWGN_clock') or (datasettype == 'all'):
                    #     samples_allEffects = clockEffects(samples_allEffects, clockeffects_dict, samp_rate, seed)
                    #
                    # if (datasettype == 'AWGN') or (datasettype == 'AWGN_clock') or (datasettype == 'all'):
                    #     samples_allEffects = phaseOffset(samples_allEffects, seed)


                    # if modulation_type.modname == 'WBFM':
                    # time.sleep(1)
                    #     import matplotlib.pyplot as plt
                    #     plt.plot(np.real(samples_channeleffect), np.imag(samples_channeleffect))
                    #     plt.show()

                    # np.random.seed(seed)
                    # start_idx = np.random.randint(transients, samples_allEffects.shape[0] - frame_length - 1)
                    # dataset[(modulation_type.modname, snr)][frame_idx, 0, :] = \
                    #     np.real(samples_allEffects[start_idx:start_idx + frame_length])
                    # dataset[(modulation_type.modname, snr)][frame_idx, 1, :] = \
                    #     np.imag(samples_allEffects[start_idx:start_idx + frame_length])
    filelocation = ""
    savefilename = filelocation + "AMC2021" + datasettype + ".01A"
    outfile1 = open(savefilename, 'wb')
    pickle.dump(dataset, outfile1)
    outfile1.close()
    # ct stores current time
    ct = datetime.datetime.now()
    print(ct)

