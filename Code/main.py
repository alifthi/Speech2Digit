from utils import utils
util = utils()
Audio ,AudioLabels = util.ReadAudio(r'/home/alifathi/Documents/AI/Git/SpeechToDigits/Data/SpokenMnist/recordings')
# util.PlotWaveForm(Audio,AudioLabels)

trainIm,trainLabel,testIm,testLabel = util.readMnist()
Spectogram = []
for i,Aud in enumerate(Audio):
    Spectogram.append(util.Audio2Spectrogram(Aud))

