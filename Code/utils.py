import glob
import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
##########################################
class utils():
    def readMnist():
        (trainIm, trainLabel), (testIm, testLabel) = ks.datasets.mnist.load_data()
        trainIm = tf.convert_to_tensor(trainIm)/255.0
        testIm = tf.convert_to_tensor(testIm)/255.0
        return [trainIm,trainLabel,testIm,testLabel]
    def ReadAudio(path):
        Audio = []
        Labels = []
        for i,file in enumerate(glob.glob(path+"\*.wav")):
            audio = tf.io.read_file(file)
            waveForm,_ = tf.audio.decode_wav(contents=audio)
            waveForm = tf.squeeze(waveForm, axis=-1)
            Audio.append(waveForm)
            Labels.append(file.split("\\")[-1].split("_")[0])
            if i%400 == 0:
                print(f'{i}th file is readed')
        return Audio,Labels
    def Audio2Spectrogram(Audio):
        Spectrogram = []
        for file in Audio:
            Spectrogram.append(ks.preprocessing.sequence.Spectrogram(file))
        return Spectrogram
    def PlotWaveForm(Audio,Label):
        from matplotlib import pyplot as plt
        rows = 3
        cols = 3
        n = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
        for i in range(n):
            t = np.random.randint(len(Audio))
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(Audio[t].numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = Label[t]
            ax.set_title(label)
    def Audio2Spectrogram(Audio):
        InputLen = 16000
        Audio = Audio[:InputLen]
        ZeroPadding = tf.zeros(16000 - np.shape(Audio)[0],
                                dtype=tf.float32)
        Audio = tf.cast(Audio, dtype=tf.float32)
        equal_length = tf.concat([Audio, ZeroPadding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram
