import numpy
#from sigprocess import audio2frame
#from sigprocess import pre_emphasis
#from sigprocess import spectrum_power
from scipy.fftpack import dct
import math

try:
    xrange(1)
except:
    xrange=range

def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    signal_length=len(signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    if signal_length<=frame_length:
      frames_num=1
    else:
      frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length)
    zeros=numpy.zeros((pad_length-signal_length,))
    pad_signal=numpy.concatenate((signal,zeros))
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T
    indices=numpy.array(indices,dtype=numpy.int32)
    frames=pad_signal[indices]
    win=numpy.tile(winfunc(frame_length),(frames_num,1))
    return frames*win
    
def spectrum_magnitude(frames,NFFT):
    complex_spectrum=numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spectrum)

def spectrum_power(frames,NFFT):
    return 1.0/NFFT * numpy.square(spectrum_magnitude(frames,NFFT))

def pre_emphasis(signal,coefficient=0.95):
   return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])

def log_spectrum_power(frames,NFFT,norm=1):

    spec_power=spectrum_power(frames,NFFT)
    spec_power[spec_power<1e-30]=1e-30
    log_spec_power=10*numpy.log10(spec_power)
    if norm:
      return log_spec_power-numpy.max(log_spec_power)
    else:
      return log_spec_power

def calcMFCC_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    #13MFCC+13 Delta+ 13Delta-Delta

    feat=calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy)
    #13MFCC+13 Delta+ 13Delta-Delta
    result1=derivate(feat)
    #13MFCC+13 Delta+ 13Delta-Delta
    result2=derivate(result1)
    #13Delta-Delta
    result3=numpy.concatenate((feat,result1),axis=1)
    result=numpy.concatenate((result3,result2),axis=1)
    return result

     
def derivate(feat,big_theta=2,cep_num=13):
   #compute delta
    result=numpy.zeros(feat.shape)
    denominator=0
    for theta in numpy.linspace(1,big_theta,big_theta):
        denominator=denominator+theta**2
    denominator=denominator*2
    for row in numpy.linspace(0,feat.shape[0]-1,feat.shape[0]):
        tmp=numpy.zeros((cep_num,))
        numerator=numpy.zeros((cep_num,)) 
        for t in numpy.linspace(1,cep_num,cep_num):
           a=0
           b=0
           s=0
           for theta in numpy.linspace(1,big_theta,big_theta):
               if (t+theta)>cep_num:
                 a=0
               else:
                   #a=feat[row][t+theta-1]
                   a=feat[int(row)][int(t+theta-1)]
               if (t-theta)<1:
                   b=0
               else:
                   #b=feat[row][t-theta-1]
                   b=feat[int(row)][int(t-theta-1)]
               s+=theta*(a-b)
           numerator[int(t-1)]=s
        tmp=numerator*1.0/denominator
        result[int(row)]=tmp
    return result
        
        
def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    feat=numpy.log(feat)
    #feat=dct(feat,type=2,axis=1,norm='ortho')[:,:cep_num]
    feat=dct(feat,type=2,axis=1,norm='ortho')[:,1:cep_num+1]
    feat=lifter(feat,cep_lifter)
    if appendEnergy:
         feat[:,0]=numpy.log(energy)
    return feat

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    
    high_freq=high_freq or samplerate/2
    signal=pre_emphasis(signal,pre_emphasis_coeff)
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate)
    spec_power=spectrum_power(frames,NFFT)
    energy=numpy.sum(spec_power,1)
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)
    feat=numpy.dot(spec_power,fb.T)
    feat=numpy.where(feat==0,numpy.finfo(float).eps,feat)
    return feat,energy
   
def log_fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    return numpy.log(feat)

def hz2mel(hz):
    return 2595*numpy.log10(1+hz/700.0)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def get_filter_banks(filters_num=20,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    mel_points=numpy.linspace(low_mel,high_mel,filters_num+2)
    hz_points=mel2hz(mel_points)
    bin=numpy.floor((NFFT+1)*hz_points/samplerate)
    fbank=numpy.zeros([filters_num,NFFT//2+1])
    for j in xrange(0,filters_num):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
  if L>0:
    nframes,ncoeff=numpy.shape(cepstra)
    n=numpy.arange(ncoeff)
    lift=1+(L/2)*numpy.sin(numpy.pi*n/L)
    return lift*cepstra
  else:
    return cepstra
