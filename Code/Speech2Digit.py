import tensorflow as tf
from tensorflow.keras import layers as ksl
import numpy as np

class Model():
    def __init__(self,modelPath,inputShape):
        self.Encoder,self.Decoder = self.ReadModel(modelPath)
        self.audioEncoder = self.BuildAudEncoder(inputShape=inputShape)
        self.net =  self.buildModel()
    def buildModel(self):
        inp1 = ksl.Input([28,28,1],name = 'imageInput')
        inp2 = ksl.Input(( 124, 129, 1),name = 'audInput')
        x1 = self.Encoder(inp1)
        x1 = self.audioEncoder(inp2)
        x1 = ksl.Flatten()(inp1)
        x2 = ksl.Flatten()(inp2)
        outIm = ksl.Dense(1024,activation = 'relu')(x1)
        outAud = ksl.Dense(1024,activation = 'relu')(x2)
        lastLayer = ksl.Lambda(self.euclidean_distance)([outIm,outAud])
        denseLayer = ksl.Dense(1, activation="sigmoid")(lastLayer)
        net = tf.keras.models.Model([inp1,inp2],denseLayer)
        return net        
    def compile(self,loss='binary_crossentropy',opt = "adam",met = ["accuracy"]):
        self.net.compile(loss = loss, optimizer=opt, metrics=met)
        self.net.summary()
    def trainModel(self,Data,labels,epochs = 10,batchSize=32):
        self.net.fit(Data,labels,epochs = epochs,batch_size = batchSize)
        pass
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
    @staticmethod
    @tf.function
    def euclidean_distance(vects):
            yA,yB = vects
            return tf.math.reduce_euclidean_norm(yA-yB,axis = 1,keepdims = True)