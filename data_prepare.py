from __future__ import print_function
import soundfile
import numpy as np
import h5py
from numpy.lib.stride_tricks import as_strided
import os


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # print(fft_length, hop_length,x.shape)
    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)
    # print(nshape, nstrides, x.shape)
    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freqs=96,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    max_freq = None
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        # ind = np.where(freqs <= max_freq)[0][-1] + 1
    # return np.transpose(np.log(pxx[:max_freqs, :] + eps))
    return np.log(pxx[:max_freqs, :] + eps)  # transpose at out function


def compute_wav_feat(filename, step, window, max_freqs=96, max_feat_len=576):
    freqs = spectrogram_from_file(filename, step=step, window=window, max_freqs=max_freqs)
    seq_len = freqs.shape[1]
    if seq_len < max_feat_len:  # if too short
        # zpad = np.zeros
        freqs = np.hstack((freqs, np.zeros((freqs.shape[0], max_feat_len - seq_len))))
    elif seq_len > max_feat_len:  # if too long
        freqs = freqs[:, (seq_len - max_feat_len) / 2:(seq_len + max_feat_len) / 2]
    ret = freqs[np.newaxis, np.newaxis, :]
    return ret


def load_data_from_one_dir(root_dir, tag=[0, 1], max_file=100):  # [0,1]:chn, [1,0]:eng
    melgrams = np.zeros((0, 1, 96, 576))
    y = []
    count = 0
    #print('root',root_dir)
    for subdir, dirs, files in os.walk(root_dir):
        #print('sub',subdir, dirs, files)
        if count > max_file:
            break
        for f in files:
            fullFilename = os.path.join(subdir, f)
            #filenameNoSuffix = os.path.splitext(fullFilename)[0]
            if f.endswith('.wav'):
                #print(f)
                count += 1
                if count % 10 == 0:
                    print(count, end='\r')
                melgram = compute_wav_feat(fullFilename, step=10, window=25, max_freqs=96, max_feat_len=576)
                melgrams = np.concatenate((melgrams, melgram), axis=0)
                y.append(tag)
                if count > max_file:
                    break

    print('onedir:', melgrams.shape, np.asarray(y).shape)
    return melgrams, np.asarray(y)

def load_h5_dataset(filename = 'ch_en_speech_dataset.h5'):
    f = h5py.File(filename, 'r')

    train_x = np.array(f['train_x']).transpose((0,2,3,1))
    train_y = np.array(f['train_y'])
    dev_x = np.array(f['dev_x']).transpose((0,2,3,1))
    dev_y = np.array(f['dev_y'])

    return train_x,train_y,dev_x,dev_y


if __name__ == '__main__':
    print('data proc')

    te = '/home/yb/sdb4t/asr_corpus/librispeech/LibriSpeech/train-clean-100'
    de = '/home/yb/sdb4t/asr_corpus/librispeech/LibriSpeech/dev-clean'
    tc = '/home/yb/sdb4t/data_thchs30/train'
    dc = '/home/yb/sdb4t/data_thchs30/dev'
    max_train_samples = 5000
    train_en_x, train_en_y = load_data_from_one_dir(te, [1, 0], max_train_samples)
    print('train_en')

    dev_en_x, dev_en_y = load_data_from_one_dir(de, [1, 0], max_train_samples*0.1)
    print('dev_en')

    train_cn_x, train_cn_y = load_data_from_one_dir(tc, [0, 1], max_train_samples)
    print('train_cn')

    dev_cn_x, dev_cn_y = load_data_from_one_dir(dc, [0, 1], max_train_samples*0.1)
    print('dev_cn')

    train_x = np.concatenate((train_en_x, train_cn_x), axis=0)
    train_y = np.concatenate((train_en_y, train_cn_y), axis=0)

    dev_x = np.concatenate((dev_en_x, dev_cn_x), axis=0)
    dev_y = np.concatenate((dev_en_y, dev_cn_y), axis=0)

    print(train_en_x.shape, train_cn_x.shape, train_en_y.shape, train_cn_y.shape)
    print(dev_en_x.shape, dev_cn_x.shape, dev_en_y.shape, dev_cn_y.shape)
    print(train_x.shape, train_y.shape, dev_x.shape, dev_y.shape)


    h5f = h5py.File('ch_en_speech_dataset_%d.h5'%max_train_samples)
    h5f.create_dataset('train_x',data=train_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('dev_x', data=dev_x)
    h5f.create_dataset('dev_y', data=dev_y)
    h5f.close()

