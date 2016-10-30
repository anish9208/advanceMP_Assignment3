
# coding: utf-8

# In[13]:

import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import glob
import os

path_to_train_audio_file = "/home/anjani/AMP/assignment_3/spk_rec_data_with_division/train"

all_spk_path = [x[0] for x in os.walk(path_to_train_audio_file)][1:]

all_speaker_features = {}

for path in all_spk_path:
    all_audio_files = glob.glob(path+"/*.wav")

    spk_audio_features = []
    
    for i in all_audio_files:
        sample_rate, X = scipy.io.wavfile.read(i)
        ceps, mspec, spec = mfcc(X)
        num_mspec = len(mspec)
        spk_audio_features.extend(mspec)
        
    all_speaker_features[path.rsplit('/',1)[1]] = spk_audio_features

print 'all speakers preprocessing done...'


# In[15]:

from sklearn.mixture import GaussianMixture

all_speaker_gmm = {}

speaker_names = []

for key in all_speaker_features:
    all_speaker_gmm[key] = GaussianMixture(covariance_type='full' , n_components= 32).fit(all_speaker_features[key])
    speaker_names.append(key)
    print "fitting "+key


# In[17]:

print speaker_names


# In[ ]:

from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# testing...

path_to_test_audio_file = "/home/anjani/AMP/assignment_3/spk_rec_data_with_division/test"

total = {}
correct = {}
incorrect = {}

for speaker in speaker_names:
    total[speaker] = 0
    correct[speaker] = 0
    incorrect[speaker] = 0

for speaker in speaker_names:
    
    audio_features = []
    all_audio_files = glob.glob(path_to_test_audio_file+"/"+speaker+"/*.wav") 
    max_match = 0 
    actual_label = speaker
    
    for i in all_audio_files:
        
        sample_rate, X = scipy.io.wavfile.read(i)
        ceps, mspec, spec = mfcc(X)
        num_mspec = len(mspec)
        audio_features.extend(mspec)
        spk_arry = []
        predict_lable = ""
        
        for one_row in audio_features :
            max_match = 0
            predicted_lable = ""
            for each_spk in all_speaker_gmm :
                score = np.exp(all_speaker_gmm[each_spk].score_samples(one_row))
                if score >= max_match:
                    max_match = score
                    predicted_lable = each_spk

            spk_arry.append(predicted_lable)
        predict_lable = max(set(spk_arry), key = spk_arry.count)

        if actual_label == predict_lable:
            print 'predicted right',lang
            correct[actual_label] = correct[actual_label]+1
        else:
            print 'predicted wrong',lang
            incorrect[actual_label] = incorrect[actual_label] + 1
            
        total[actual_label] += 1

print correct
print incorrect
print total


# In[9]:

tot_correct = 0
tot_incorrect = 0

for lang in language_names:
    tot_correct += correct[lang]
    tot_incorrect += incorrect[lang]
accuracy = (1.0 * tot_correct)/(tot_correct + tot_incorrect)

print 'accuracy is : ',accuracy

