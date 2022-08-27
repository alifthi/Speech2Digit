# Speech2Digit

In this project i create a model that generate an image that says in a speech.

# Generator

As generator in this project i used a Convolutional Auto Encoder (CAE) that i trained as image compression project. 

# Method description

Idea for this project is that, try to encode the audio spectrom and image the same.

To achive this goal i use an auto encoder network then i disassemble just encoder part from auto encoder and i freeze it to not trained any more, then i create a new encoder for audio spectrom then  train audio encoder with this goal to reduce euclidean distance bitween outputs of audio encoder and image encoder.

That means i try to train model to encode audio just same as image encoding.

After that and when  audio encoder trained and  euclidean distance between outputs of both encoders reduced i will assemble encoder of audio this decoder of auto encoder.
