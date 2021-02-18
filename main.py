'''
This file serves as the api to receive 
the file from the user-end as http request
sends the segmentation results as http response.
'''
import os
import web
import random
import json
import wave
#import umm_seg
import numpy as np
#import silence_classf
import csv
#import librosa
import torch
import pickle
import time
import scipy.io.wavfile as wav
from classify_model import classify_one_cycle, RNN_model
from calcmfcc import calcMFCC_delta_delta

# the full path to current dir, this is required to load the assets
dir_=os.path.abspath('.')+'/'
    
urls = (
    '/getaudio' , 'get_audio_file',
    '/curr_audio.wav' , 'send_audio_curr',
    '/sample_1.wav' , 'send_audio_sample1',
    '/sample_2.wav' , 'send_audio_sample2',
    '/sample_3.wav' , 'send_audio_sample3',
    '/sample_4.wav' , 'send_audio_sample4',
    '/getsample' , 'get_sample_file'
)

app = web.application(urls, globals())

class get_sample_file:
    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return "GET Success"

    def POST(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')
        data=web.input()

        start_time=[]
        end_time=[]
        sil_start_time=[]
        sil_end_time=[]

        if data.type=='audio/mpeg':
            wf=open(dir_+'storage/'+str(data.url)[-6:]+'.mp3','wb')
            wf.write(data.file)
            wf.close()

        elif data.type=='audio/wav':
            #read sample and create a copy
            sample_name=data.filename
            sf=open(dir_+'storage/'+sample_name,'rb')
            wav_bytes=sf.read()

            file_name='curr.wav'
            wf=open(dir_+'storage/'+file_name,'wb')
            wf.write(wav_bytes)
            wf.close()
            print ("audio received ...")

            # make single channel [uncomment if passing dual channel audio]
            #silence_classf.make_one_channel(dir_+'storage',file_name)

            # call voice separation required coz due to ambient noise the silence detection doesn't work properly
            #new_fn_bg,new_fn_fg=silence_classf.vocal_separation(dir_+'storage',file_name)
            #sil_file_name=new_fn_fg

            # call umm segmentation
            (rate,sig) = wav.read(dir_+'storage/'+file_name)
            start_time = time.time()
            mfcc_feat = calcMFCC_delta_delta(sig,rate)
            mid_time = time.time()
            print("Signal processing time:{} s".format(mid_time-start_time))

            model = RNN_model(15)
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
            with open("models/label_person_mapping_new.pickle", "rb") as file:  
                label_person_mapping = pickle.load(file)
            label2person = label_person_mapping["label2person"]
            output_prob, output_label, output_person = classify_one_cycle(model, mfcc_feat, label2person)
            end_time = time.time()
            target_prob = output_prob[output_label]
            if "_" in output_person:
                output_person = output_person.split("_")[0] + " " + output_person.split("_")[1]
            print("output_person:{}".format(output_person))
            print("output_probabilities:{}".format(target_prob))
            print("Model predicting time:{} s".format(end_time-mid_time))
            
            return json.dumps({'msg':"Your file is uploaded!",
            'output_person':output_person, "output_probability": round(target_prob,4),
            "total_time": round(end_time - start_time, 3),
            })

    def OPTIONS(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')
        return "OPTIONS Success"

class send_audio_curr:

    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return open(dir_+'storage/ano_new_curr.wav','rb')

class send_audio_sample1:

    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return open(dir_+'storage/Qiaomu.wav','rb')

class send_audio_sample2:

    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return open(dir_+'storage/Qiaomu.wav','rb')

class send_audio_sample3:

    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return open(dir_+'storage/Qiaomu.wav','rb')

class send_audio_sample4:

    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return open(dir_+'storage/Recorded.wav','rb')


class get_audio_file:


    def GET(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')

        return "GET Success"

    def POST(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')
        data=web.input()

        start_time=[]
        end_time=[]
        sil_start_time=[]
        sil_end_time=[]

        if data.type=='audio/mpeg':
            wf=open(dir_+'storage/'+str(data.url)[-6:]+'.mp3','wb')
            wf.write(data.file)
            wf.close()

        elif data.type=='audio/wav':

            print("\n\n\n",data.url)
            file_name="curr.wav"
        
            print("Trying to load now")
            #file_name='curr.wav'
            wf=open(dir_+'storage/'+ file_name,'wb')
            wf.write(data.file)
            wf.close()
            print ("audio received ...")

            # make single channel [uncomment if passing dual channel audio]
            #silence_classf.make_one_channel(dir_+'storage',file_name)

            # call voice separation required coz due to ambient noise the silence detection doesn't work properly
            #new_fn_bg,new_fn_fg=silence_classf.vocal_separation(dir_+'storage',file_name)
            #sil_file_name=new_fn_fg

            (rate,sig) = wav.read(dir_+'storage/'+file_name)
            start_time = time.time()
            mfcc_feat = calcMFCC_delta_delta(sig,rate)
            mid_time = time.time()
            print("Signal processing time:{} s".format(mid_time-start_time))

            model = RNN_model(15)
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
            with open("models/label_person_mapping_new.pickle", "rb") as file:
                label_person_mapping = pickle.load(file)
            label2person = label_person_mapping["label2person"]
            output_prob, output_label, output_person = classify_one_cycle(model, mfcc_feat, label2person)
            end_time = time.time()
            target_prob = output_prob[output_label]
            output_person = output_person.split("_")[0] + " " + output_person.split("_")[1]
            print("output_person:{}".format(output_person))
            print("output_probabilities:{}".format(target_prob))
            print("Model predicting time:{} s".format(end_time-mid_time))
            
            return json.dumps({'msg':"Your file is uploaded!",
            'output_person':output_person, "output_probability": round(target_prob,4), 
            "total_time": round(end_time - start_time,3),
            })

    def OPTIONS(self):
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        web.header('Access-Control-Allow-Methods',  'POST, GET, OPTIONS')
        return "OPTIONS Success"


if __name__ == "__main__":
    app.run()
