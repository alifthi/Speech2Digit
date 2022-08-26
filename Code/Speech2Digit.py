import tensorflow as tf
from tensorflow.keras import layers as ksl
from keras import backend as K
import numpy as np

class Model():
    def __init__(self,modelPath,inputShape):
        self.Encoder,self.Decoder = self.ReadModel(modelPath)
        self.audioEncoder = self.BuildAudEncoder(inputShape=inputShape)
        self.net =  self.buildModel()
    def buildModel(self):
        inp1 = ksl.Input([28,28,1],name = 'imageInput')
        inp2 = ksl.Input(( 124, 129, 1),name = 'audInput')
        x1 = ksl.Flatten()(inp1)
        x2 = ksl.Flatten()(inp2)
        sharedDense = ksl.Dense(1024,activation = 'relu')
        x1 = sharedDense(x1)
        x2 = sharedDense(x2)
        outIm = self.Encoder(x1)
        outAud = self.audioEncoder(x2)

        lastLayer = ksl.Lambda(self.euclidean_distance)([outAud,outIm])
        denseLayer = ksl.Dense(1, activation="sigmoid")(lastLayer)
        net = tf.keras.models.Model([inp1,inp2],denseLayer)
        return net        
    def compile(self,loss='binary_crossentropy',opt = "adam",met = ["accuracy"]):
        self.net.compile(loss = loss, optimizer=opt, metrics=met)
        self.net.summary()
    @staticmethod
    def BuildAudEncoder(inputShape):
        Input = tf.keras.Input(shape=inputShape)
        x = ksl.Conv2D(16, kernel_size=3, strides=3, padding='same')(Input)
        x = ksl.BatchNormalization()(x)
        x = ksl.Conv2D(32, kernel_size=3, strides=3, padding='same')(x)
        x = ksl.BatchNormalization()(x)
        x = ksl.Dense(64,activation='relu')(x)
        return tf.keras.models.Model(Input,x)
    @staticmethod
    def ReadModel(path):
        BaseModel = tf.keras.models.load_model(path)
        Encoder = BaseModel.layers[0]
        Decoder = BaseModel.layers[1]
        for i in Encoder.layers:
            i.trainable = False
        for j in Decoder.layers:
            j.trainable = False
        return [Encoder,Decoder]
#     @tf.function
    def euclidean_distance(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))