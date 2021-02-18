# User_Identification_with_Breath
A system with an RNN model for user identification with Breath signals along with a web app user interface

### Requirements
The program was runned with python 3.7, required packages:
* web.py
* numpy
* scipy
* torch

### Run the demo
To run the webapp, first click the index.html, then run `python main.py 7000` (7000 is our specified port number of the server). It is recommended to run on a device with gpu to accelerate the running of RNN model.<br>

You can click the sample files to show the results. The original sample files are the audio breath files from different users, but these files have all been replaced to the breath file of my own due to privacy concern. The last one loads my online-recorded breath audio saved in the "storage" folder. It is recorded online from laptop, so it outputs with a low probability score. You can also record your own breath cycle with the "Record" button, name it as "Recorded.wav", and put it under the "storage" folder, then you can classify your own identity by clicking the "From Recording" after training your own model.<br>

You can also upload a .wav breath file from the "storage" folder by clicking the "Choose a file" button. The web app takes in a wav file, convert it to MFCC features with the `calcmfcc.py`, and output the user identity and confidence of the classification. The web app will display the waveform of the breath and you can also play the breath file by clicking the "play" button. <br>

Note: We have only implemented the user identification for users within the registered group. Sorry but identification for users outside the group is not available.

### Train your own model
The `Classification.py` file in the "Train_model" folder contains the code for training the RNN model and evaluating its performance. The model weights are stored in the "models" folder for classification of users participating in our project. You can modify this file to train your own model. <br> 

The "js" folder contains the javascript file for web app development. The code for web app development is modified based on the code from [Disfluency-Removal-API](https://github.com/sagniklp/Disfluency-Removal-API), sincere thankfulness to the authors.
