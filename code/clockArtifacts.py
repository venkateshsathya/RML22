import numpy as np
import scipy.interpolate as interpolate
def CFOArtifact(samples, XO_val, clockeffects_dict, samp_rate):
    Ts = 1 / samp_rate  # sample interval
    samples_length = samples.shape[0]
    n = np.arange(0, samples_length)
    CFO_val = XO_val*clockeffects_dict['LOScalingFactor']
    CFO_val = CFO_val[0:samples_length]
    CFO_mult_n = np.multiply(CFO_val, n)

    exp_val = np.exp(1j * 2 * np.pi * CFO_mult_n * Ts)
    samples_withCFO = np.multiply(exp_val, samples)
    return samples_withCFO

def SROArtifact(samples, XO_val, clockeffects_dict, samp_rate):
    Ts = 1 / samp_rate  # Sample interval
    samples_length = samples.shape[0] # Total number of samples
    f_ABW = clockeffects_dict['XOFreq']*clockeffects_dict['TimetickScalingFactor'] # analog bandwidth
    SRO_val_full = XO_val*clockeffects_dict['TimetickScalingFactor']
    SRO_val = SRO_val_full[0:samples_length]
    realval = np.real(samples)
    imagval = np.imag(samples)



    Ts_range = np.linspace(0, samples_length * Ts, samples_length)  # np.arange(0, samples_length * Ts, Ts)
    Ts_prime_range =  Ts_range +((Ts*clockeffects_dict['TimetickScalingFactor'])/f_ABW)*SRO_val #np.cumsum(Ts_prime)
    f_imag = interpolate.interp1d(Ts_range, imagval, fill_value='extrapolate')
    f_real = interpolate.interp1d(Ts_range, realval, fill_value='extrapolate')
    samples_SRO_real = f_real(Ts_prime_range)
    samples_SRO_imag = f_imag(Ts_prime_range)
    samples_SRO = samples_SRO_real + 1j * samples_SRO_imag
    return samples_SRO

def phaseOffset(samples, seed):
    # theta = np.random.uniform(0,2*np.pi)
    np.random.seed(seed)  # setting the random seed
    # To check if the seed has been set, use the command np.random.get_state()[1][0]
    phaseOffsetVal = np.random.uniform(0, 2 * np.pi)
    samples_phaseOffset = np.multiply(samples, np.exp(1j * phaseOffsetVal))
    return samples_phaseOffset