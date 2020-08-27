#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser
# import sys
# sys.path.append('./model/modelv2')
# from utils import *
from deepspeech import Model#, printVersions
import os
import time
import numpy as np
import librosa
import scipy
from tqdm import tqdm
import subprocess
#############Voice-To-Text#############

#####recording parameters
import pyaudio
CHUNK = 1024 #1024
FORMAT = pyaudio.paInt16
# try:
#     CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
#     #2
# except:
#     print("No sound channel configured. Set CHANNEL = 1")
#     CHANNELS = 1
CHANNELS = 1
RATE = 16000 # 44100
EXTRA_SECONDS = 1.0
RECORD_SECONDS = 5 + EXTRA_SECONDS
BACKGROUND_RECORD_SECONDS = 2

#### denoising functions
def _stft(x, nperseg=400, noverlap=239, nfft=1023):
    """
    Get STFT using scipy.signal.stft.

    x: audio data as in array.
    nperseg, noverlap, nfft: argument for scipy.signal.stft
    """
    _, _, Z = scipy.signal.stft(x, window="hamming",
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   nfft=nfft)
    assert Z.shape[0] == 512
    return np.array(Z)

def _istft(x, nperseg=400, noverlap=239, nfft=1023):
    """
    Get the inverse STFT using scipy.signal.istft.

    nperseg, noverlap, nfft: argument for scipy.signal.istft
    """
    _, Z = scipy.signal.istft(x, window="hamming",
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   nfft=nfft)
    return np.array(Z)

def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def removeNoise(
    audio_data,
    noise_data,
    #nperseg=400, noverlap=239, nfft=1023
    n_grad_freq=2,
    n_grad_time=4,
#     n_fft=2048,
#     n_fft=1023,
#     win_length=2048,
#     hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_data (array): The first parameter.
        noise_data (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted
    """
#     if verbose:
#         start = time.time()
    ## STFT over noise
    noise_stft = _stft(noise_data)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    ## Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
#     if verbose:
#         print("STFT on noise:", td(seconds=time.time() - start))
#         start = time.time()
    ## STFT over signal
#     if verbose:
#         start = time.time()
    sig_stft = _stft(audio_data)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
#     if verbose:
#         print("STFT on signal:", td(seconds=time.time() - start))
#         start = time.time()
    ## Calculate value to mask dB to
    mask_gain_dB = np.min(sig_stft_db)
#     print("Noise threshold, Mask gain dB:\n",noise_thresh, mask_gain_dB)
    ## Create a smoothing filter for the mask in time and frequency
    filter_compt = np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1]
    smoothing_filter = np.outer(
            filter_compt,
            filter_compt,
        )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    ## calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    ## mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
#     if verbose:
#         print("Masking:", td(seconds=time.time() - start))
#         start = time.time()
    ## convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
#     if verbose:
#         print("Mask convolution:", td(seconds=time.time() - start))
#         start = time.time()
    ## mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
#     if verbose:
#         print("Mask application:", td(seconds=time.time() - start))
#         start = time.time()
    ## recover the signal
    recovered_signal = _istft(sig_stft_amp)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal)))
    return recovered_signal.astype('float32') #audio data as if loaded from librosa.load
# return sig_stft_amp



def record_and_denoise( enroll = False, phrase = '', sample_phrase_list = [], RECORD_SECONDS = RECORD_SECONDS):
    """
    Record voice and denoise using removeNoise function.

    enroll: whether it is for enrollment or not.
    phrase: pass the phrase the user provided. If empty, phrase will be transcribed.
    sample_phrase_list: a list of sample phrases.
    RECORD_SECONDS: time to record in seconds.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    print()
    print(" Speak your query:\n")
    print(" Recording {} seconds \n".format(RECORD_SECONDS - EXTRA_SECONDS))
    if enroll:input(' Ready to start? (press enter)')
    else: print(" Recording starts soon...\n")#time.sleep(1)
    frames_bg = []
    for i in range(0, int(RATE / CHUNK * (BACKGROUND_RECORD_SECONDS) ) ):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames_bg.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    print(" Recording starts in 3 second...")
    time.sleep(2)   # start 1 second earlier
    frames = []
    print(" Speak now!")
    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(" Recording complete.")
    audio_data = (np.frombuffer(b''.join(frames), dtype=np.int16)/32767)
    bg_data = (np.frombuffer(b''.join(frames_bg), dtype=np.int16)/32767)
    # denoised_data = removeNoise(audio_data, bg_data)#.astype('float32')
    return audio_data #denoised_data


#######Deepspeech Voice-To-Text Parameters########
DS_FOLDER = 'deepspeech_data'
if not os.path.exists(DS_FOLDER):
    os.mkdir(DS_FOLDER)
DS_model_file_path = 'deepspeech_data/deepspeech-0.7.4-models.pbmm'
beam_width = 500
DS_model = Model(DS_model_file_path)
DS_model.setBeamWidth(beam_width)
DS_model.enableExternalScorer('deepspeech_data/deepspeech-0.7.4-models.scorer')

def get_text(data, model = DS_model):
    """
    Transcribe text from audio.

    data: audio data as in array read from librosa with sampling rate 16000.
    model: Deepspeech ASR model.
    """
#     y , s = librosa.load(fpath, sr=16000)
    y = (data* 32767).astype('int16')
    text = model.stt(y)
    return text


def get_query( file = '', phrase = ''):
    """
    returns an embedding vector and denoised audio data array.

    file: path to the audio file
        if given, speaker's audio is read from 'file'.
            Miminum of either NOISE_DURATION_FROM_FILE or the first two seconds (RATE*2) will be considered as background noise.
        if not given, invoke record_and_denoise function.
    enroll: indicate whether the user is enrolling or not.
    phrase: phrase is passed if the user provide it. Otherwise pass '' and it will be transcribed later.
    """
    if file:
        data , _ = librosa.load(file,sr=RATE)
        NOISE_DURATION_FROM_FILE = int(len(data)*0.25) # N_D_F_F in terms of lenth of data not second
        NOISE_DURATION_FROM_FILE = min(NOISE_DURATION_FROM_FILE, RATE*2)
        noise, data = np.split(data,[NOISE_DURATION_FROM_FILE])
        denoised_data = removeNoise(data,noise)
    else:
        denoised_data = record_and_denoise()
    query = get_text(denoised_data)
    return query


def main():
    running = True
    file = ''
    while running:
        args = input("\n Please type file path or hit enter to record a query:").lower()
        print()
        if args:
            file = args
        query = get_query(file = file)
        print(f"\n Query read: {query}")
        while True:
            var = input(f"\n Use this query?(y/n):").lower()
            if var == 'y' or var == 'yes':
                running = False
                break
            elif var == 'n' or var == 'no':
                running = True
                break
            else: continue
    return query
if __name__ == "__main__":
    query = main()
    subprocess.call(['python','nl2sql.py', '-q',query])
