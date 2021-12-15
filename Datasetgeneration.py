import sys
from transmitters import transmitters
from source_alphabet import source_alphabet
from clockArtifacts import *
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
import math


######### INPUT PARAMETERS #####################
# AI: potentially move this to a JSON file


## General signal parameters
f_c = 1e9 # Center frequency
clockrate = 100e6 # assuming usage of USRP N310, assuming a ADC and DAC clock rate equal to max analog bandwidth.
audio_rate = 44.1e3
samples_per_symbol = 2 #Upsampling factor
Tsym_LTE = 71.3e-6# Tsym for LTE is 71.3 microseconds.
# Samp_rate caluclated as = 8/71.3 (8 is the upsampling factor - i.e. Number of samples per symbol)
samp_rate = samples_per_symbol/Tsym_LTE#112.2e3
Ts = 1 / samp_rate  # sample interval

# Modulation parameters
excess_bandwidth = 0.35
modulation_index_CPFSK = 0.5
bandwidth_time_product = 0.3 # source for BT value chosen: https://comblock.com/download/com1028.pdf
sensitivity_GFSK = 1.57 # approx to pi/2.
output_stream_rate_WBFM = math.ceil(samp_rate/audio_rate)*audio_rate
max_freq_dev_WBFM = 75e3 # wideband FM freq deviation typical value - chosen from https://en.wikipedia.org/wiki/Frequency_modulation
tau=75e-6 # preemphasis time constant (default 75e-6), value used frorm https://github.com/gnuradio/gnuradio/blob/master/gr-analog/python/analog/wfm_tx.py


# Dataset simulation parameters
dataset = {}
if len(sys.argv) < 2:
    numFrames_permodsnr = 1000  # Number of frames per modulation type and SNR.
else:
    numFrames_permodsnr = int(sys.argv[1])
frame_length = 128  # length of a frame.
transients = 1000  # typically after channel, clock effects are applied, the output has trasients for
# the first 500 odd samples that need to be ignored.
# In a long array fo samples post effects such as clock,AWGN,fading etc., we iteratively carve chunks of samples equal to frame_length.
# and jump by an increment that is decided by the samples_incrementscale.
samples_incrementscale = 0.05
# The audio source file has almost no signal for the first few seconds.
analog_transients = int(audio_rate * 5)
# Samples of digitalsample_len will be passed through modulation, AWGN, ClockEffects, Fading blocks.
# A single frame will be carved out with a randomly placed window,
# with a minimum offset of transients out of signal of length digitalsample_len.
num_IF_perstream = 50 # A stream of this many frames are created and IFs carved out. Therefore the lower this value is, more likely the IFs will be more indepenedent of each other, since for each stream a separate instantiation of artifacts are initiated.
digitalsample_len = frame_length * num_IF_perstream + transients


""""
clean - only the modulated signal with pulse shaping whereever applicable
AWGNOnly - Apply additive thermal noise applied to clean modulated signal for specified SNR levels, to the modulated signal.
Clock Only - Apply errors due to clock drifts - center frequency offset (CFO) and sample rate offset (SRO) and phase offset to the modulated signal.
FadingOnly - Apply fading effects only to the modulation signals.
All - All effects in the order of SRO->Fading->CFO->Phase offset-> AWGN
"""
datasettype_list = ['clean','AWGNOnly','ClockOnly','FadingOnly','All'] #['clean’,’AWGNOnly’,’ClockOnly’,’FadingOnly’,’All']
# 


## Signal artifact parameters

#AWGN
snr_levels = range(-20, 21, 2)

#Fading
# delays in nanoseconds converted into fractional sample delays. Ech sample delay Ts = 9 micro seconds approx.
delays = [val_temp * (1e-9) * samp_rate for val_temp in [0, 50, 120, 200, 230, 500, 1600, 2300, 5000]]
mags_dB = [-1, -1, -1, 0, 0, 0, -3, -5, -7]
mags = [10 ** (mags_val / 20.0) for mags_val in mags_dB]
fD = 70  # ETU70, max doppler ferquency/frequency devaition - 70Hz.
ntaps = 8
numSinusoids = 8
LOS = False  # Rayleigh channel, no LOS component
Kfactor = 4

## Clock effects
clockeffects_dict = {}
clockeffects_dict['XOFreq'] = 10e6
clockeffects_dict['XO_standardDeviation'] = 1e-4
clockeffects_dict['XO_maxdeviation'] = 5
clockeffects_dict['LOScalingFactor'] = f_c/clockeffects_dict['XOFreq']
clockeffects_dict['TimetickScalingFactor'] = clockrate/clockeffects_dict['XOFreq']
clockeffects_dict['CFO_standardDeviation'] = clockeffects_dict['XO_standardDeviation']*clockeffects_dict['LOScalingFactor']
clockeffects_dict['SRO_standardDeviation'] = clockeffects_dict['XO_standardDeviation']*clockeffects_dict['TimetickScalingFactor']
clockeffects_dict['CFO_maxdeviation'] = clockeffects_dict['XO_maxdeviation']*clockeffects_dict['LOScalingFactor']
clockeffects_dict['SRO_maxdeviation'] = clockeffects_dict['XO_maxdeviation']*clockeffects_dict['TimetickScalingFactor']






# Analog data is large and takes time to read. Therefore done once in the beginning and random windows
# used to extract samples for each input frame.
# NOTE: If there are issues loading the large audio file such as RAM being filled up,
# you can use any smaller sized audio file.

print("Loading analog samples as an one time operation. Please be patient.")
tb = gr.top_block()
src_cont = source_alphabet("continuous")
snk_cont = blocks.vector_sink_f()
tb.connect(src_cont, snk_cont)
tb.run()

analogdata = np.array(snk_cont.data(), dtype=np.complex64)
len_analogdata = analogdata.shape[0]
print("Finished loading analog samples.")

seed = 0

print("changed datasettype_list to all")
for datasettype in datasettype_list:#datasettype_list:
    print("Datasettype is :", datasettype)
    for modulation_flavour in list(transmitters.keys()):
        for modulation_type, snr in product(list(transmitters[modulation_flavour]), snr_levels):
            dataset[(modulation_type.modname, snr)] = np.zeros([numFrames_permodsnr, 2, frame_length], dtype=np.float32)
            ct = datetime.datetime.now()
            print("Generating data for Modulation type: ", modulation_type.modname," and snr: ", snr, " at time ", ct)
            frame_idx = 0
            noise_amp = 10 ** (-snr / 20.0)  # you need to divide by 20 instead of 10 to get the correct SNR.
            while frame_idx < numFrames_permodsnr:
                # print(modulation_flavour,modulation_type,snr,frame_idx)

                np.random.seed(seed)  # setting the random seed
                # To check if the seed has been set, use the command np.random.get_state()[1][0]

                if modulation_type.modname == 'GFSK':
                    mod = modulation_type(samples_per_symbol,sensitivity_GFSK,bandwidth_time_product)
                elif modulation_type.modname == 'CPFSK':
                    mod = modulation_type(modulation_index_CPFSK,samples_per_symbol)
                elif modulation_type.modname == 'WBFM':
                    mod = modulation_type(audio_rate, output_stream_rate_WBFM, tau,max_freq_dev_WBFM)
                elif (modulation_type.modname == 'AM-DSB') or (modulation_type == 'AM-SSB'):
                    mod = modulation_type(audio_rate, samp_rate)
                else:
                    mod = modulation_type(samples_per_symbol, excess_bandwidth)  # in the modified transmitter file, change
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
                    # Bock signature:  gnuradio.blocks.vector_source_f(data, repeat = False, vlen = 1, tags)


                ## XO value generation. This value will be passed onto SRO and CFO blocks where this will be scaled.
                ferr_bias_XO = np.random.uniform(-clockeffects_dict['XO_maxdeviation'] + clockeffects_dict['XO_standardDeviation'],\
                                                 clockeffects_dict['XO_maxdeviation'] - clockeffects_dict['XO_standardDeviation'])
                # digitalsample_len is the length of input source. This is modulated and upsampled. Therefore the XO value length that is applied to the modulated upsampled symbols should be of appropriate length of the upsampled modulated samples.
                XO_val_len = digitalsample_len*samples_per_symbol+10
                XO_val = np.zeros((XO_val_len,))
                XO_val[0] = clockeffects_dict['XO_standardDeviation'] * np.random.randn(1, ) + ferr_bias_XO
                # Check to ensure that CFO is contained within maximum deviation.
                while (XO_val[0] > clockeffects_dict['XO_maxdeviation']) or (XO_val[0] < -clockeffects_dict['XO_maxdeviation']):
                    XO_val[0] = clockeffects_dict['XO_standardDeviation'] * np.random.randn(1, ) + ferr_bias_XO

                for i in range(1, XO_val_len):
                    XO_val[i] = clockeffects_dict['XO_standardDeviation'] * np.random.randn(1, ) + XO_val[i - 1]
                    # Check to ensure that CFO is contained within maximum deviation.
                    while (XO_val[i] > clockeffects_dict['XO_maxdeviation']) or (XO_val[i] < -clockeffects_dict['XO_maxdeviation']):
                        XO_val[i] = clockeffects_dict['XO_standardDeviation'] * np.random.randn(1, ) + XO_val[i - 1]


                if datasettype == 'clean':
                    tb.connect(src, mod, snk)
                    tb.run()
                    samples = np.array(snk.data(), dtype=np.complex64)
                elif datasettype == 'AWGNOnly':
                    tb.connect(src, mod)
                    tb.connect(noise_block, (add_block, 1))
                    tb.connect(mod, (add_block, 0))
                    tb.connect(add_block, snk)
                    tb.run()
                    samples = np.array(snk.data(), dtype=np.complex64)
                elif datasettype == 'ClockOnly':
                    # Apply SRO followed by CFO followed by phase offset
                    tb.connect(src, mod, snk)
                    tb.run()
                    samples_mod = np.array(snk.data(), dtype=np.complex64)
                    samples_SRO = SROArtifact(samples_mod, XO_val, clockeffects_dict, samp_rate)
                    samples_SRO_CFO = CFOArtifact(samples_SRO, XO_val, clockeffects_dict, samp_rate)
                    samples_allClockArtifacts = phaseOffset(samples_SRO_CFO, seed)
                    samples = samples_allClockArtifacts
                elif datasettype == 'FadingOnly':
                    # Apply fading only
                    tb.connect(src, mod, fading_block, snk)
                    tb.run()
                    samples = np.array(snk.data(), dtype=np.complex64)
                elif datasettype == 'All':
                    tb.connect(src, mod, snk)
                    tb.run()
                    samples_clean = np.array(snk.data(), dtype=np.complex64)
                    samples_SRO = SROArtifact(samples_clean, XO_val, clockeffects_dict, samp_rate)
                    samples_SRO_src_block = blocks.vector_source_c(samples_SRO, False, 1, [])
                    snk2 = blocks.vector_sink_c()
                    tb.connect(samples_SRO_src_block, fading_block, snk2)
                    tb.run()
                    samples_SRO_Fading = np.array(snk2.data(), dtype=np.complex64)
                    samples_SRO_Fading_CFO = CFOArtifact(samples_SRO_Fading, XO_val, clockeffects_dict, samp_rate)
                    samples_SRO_Fading_CFO_Phaseoffset = phaseOffset(samples_SRO_Fading_CFO, seed)

                    samples_SRO_Fading_CFO_Phaseoffset_block = blocks.vector_source_c(samples_SRO_Fading_CFO_Phaseoffset, False, 1, [])
                    # tb.connect(src, mod)
                    snk3 = blocks.vector_sink_c()
                    tb.connect(noise_block, (add_block, 1))
                    tb.connect(samples_SRO_Fading_CFO_Phaseoffset_block, (add_block, 0))
                    tb.connect(add_block, snk3)
                    tb.run()
                    samples_alleffects = np.array(snk3.data(), dtype=np.complex64)
                    samples = samples_alleffects

                # remove transients from the samples post fading.
                samples = samples[transients:]
                len_sampAlleffects = samples.shape[0]

                np.random.seed(seed)
                incr_idx = np.random.randint(0,int(len_sampAlleffects*samples_incrementscale))
                start_idx = 0
                start_idx = start_idx + incr_idx
                #frame_idx = frame_idx + 1
                while (start_idx +frame_length < len_sampAlleffects) and (frame_idx < numFrames_permodsnr):
                    frame_allEffects = samples[start_idx:start_idx+frame_length]
                    dataset[(modulation_type.modname, snr)][frame_idx, 0, :] = np.real(frame_allEffects)
                    dataset[(modulation_type.modname, snr)][frame_idx, 1, :] = np.imag(frame_allEffects)
                    seed = seed + 1
                    np.random.seed(seed)
                    incr_idx = np.random.randint(0,int(len_sampAlleffects*samples_incrementscale))
                    start_idx = start_idx + incr_idx + frame_length
                    #print(start_idx,frame_idx)
                    frame_idx = frame_idx + 1
#                     print(frame_idx,seed)

                #frame_idx = frame_idx - 1
#                 seed = seed - 1
#                 print("Outside while loop")
#                 print(frame_idx,seed)

    filelocation = ""
    savefilename = filelocation + "AMC2021" + datasettype + ".01A"
    outfile1 = open(savefilename, 'wb')
    pickle.dump(dataset, outfile1)
    outfile1.close()
    # ct stores current time
    ct = datetime.datetime.now()
    print(ct)
#     outfile1 = open(savefilename + '_Params.txt', 'at')
#     outfile1.write(str(TrainParams))
#     outfile1.write("\n \n")
#     outfile1.write(str(DatasetParams))
#     #pickle.dump(dataset, outfile1)
#     outfile1.close()
