import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torchaudio
import scipy.signal

epsilon = np.finfo(float).eps
SR = 16000   # sampling rate

# Define parameters
n_fft = 512
hop_length = 160
win_length = 512
window = torch.from_numpy(scipy.signal.windows.hamming(win_length)).float()

def get_filepaths(directory, ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)


def make_spectrum(filename=None, y=None, feature_type='logmag', _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = torchaudio.load(filename)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False, return_complex=True).squeeze()
    amplitude = torch.abs(stft)
    log_amp = torch.log1p(amplitude)
    phase = torch.angle(stft)

    return log_amp, phase, len(y)


if __name__ == '__main__':
    target_root = "/Users/huan/Downloads/TIMIT_speakers"

    for stage in ['train', 'test']:
        print(f'converting {stage} files')
        train_path = os.path.join(target_root, stage)
        train_convert_save_path = os.path.join(target_root + '_pt', stage)

        n_frame = 128
        wav_files = get_filepaths(train_path)
        for wav_file in tqdm(wav_files):
            wav, sr = torchaudio.load(wav_file)
            out_path = wav_file.replace(train_path, train_convert_save_path).split('.w')[0]
            output, _, _ = make_spectrum(y=wav)
            for i in np.arange(output.shape[1] // n_frame):
                Path(out_path).mkdir(parents=True, exist_ok=True)
                out_name = out_path + '_' + str(i) + '.pt'
                split_output = output[:, i * n_frame:(i + 1) * n_frame]
                torch.save(split_output, out_name)

                # plt.figure()
                # plt.imshow(split_output.unsqueeze(-1))
                # plt.gca().invert_yaxis()
                # plt.title(out_path)
                # plt.show()
