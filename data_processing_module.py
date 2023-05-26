import librosa
import librosa.display
import matplotlib.pyplot as plot
import matplotlib
import yaml
%matplotlib tk


class DataProcessingModule:
    def __init__(self,config):
            self.y = config['data_processing_module']

    def mfcc_method(self, audio, sample_rate):
        mfcc_values = librosa.feature.mfcc(y = audio, sr = sample_rate,
                                           S = None, n_mfcc = self.y['n_mfcc'],
                                           dct_type = self.y['dct_type'],
                                           norm = 'ortho', n_fft = self.y['n_fft'],
                                           hop_length = self.y['hop_length'],
                                           power = self.y['power'], htk = self.y['htk'],
                                           n_mels = self.y['n_mels'],fmin = self.y['fmin'],
                                           fmax = None)
        return mfcc_values 

    def filter_bank_method(self,audio,sample_rate):
        filter_bank_values = librosa.feature.melspectrogram(y = audio, sr = sample_rate,
                                                            S = None, n_fft = self.y['n_fft'],
                                                            hop_length = self.y['hop_length'],
                                                            power = self.y['power'], htk = self.y['htk'],
                                                            norm = 1, n_mels = self.y['n_mels'],
                                                            fmin = self.y['fmin'], fmax = None)
        return filter_bank_values

if __name__ == '__main__':
    spectogram = DataProcessingModule()
    pop1_path = 'Desktop/pwr-training/signals/pop2.wav'
    audio, sample_rate = librosa.load(pop1_path)
    spectogram_mfcc = spectogram.mfcc_method(audio, sample_rate)
    spectogram_fb = spectogram.filter_bank_method(audio, sample_rate)
    plot.figure()
    #Displaying MFCC
    plot.subplot(3, 1, 1)
    plot.title('MFCC')
    librosa.display.specshow(spectogram_mfcc, fmax=4096, x_axis='time')
    plot.colorbar()
    #Displaying Filter Bank
    plot.subplot(3, 1, 3)
    plot.title('Filter Bank')
    librosa.display.specshow(spectogram_fb, y_axis='mel', fmax=4096, x_axis='time')
    plot.colorbar(format = '%+2.0f dB')
    plot.show()
    