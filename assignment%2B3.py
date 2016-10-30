
# coding: utf-8

# In[100]:

import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import glob
import os

path_to_train_audio_file = "/home/anjani/AMP/assignment_3/lang_rec_data_with_division/train"

all_lang_path = [x[0] for x in os.walk(path_to_train_audio_file)][1:]

all_language_features = {}

for path in all_lang_path:
    all_audio_files = glob.glob(path+"/*.wav")

    lang_audio_features = []
    
    for i in all_audio_files:
        sample_rate, X = scipy.io.wavfile.read(i)
        ceps, mspec, spec = mfcc(X)
        num_mspec = len(mspec)
        if path.rsplit('/',1)[1] == 'odia' :
            lang_audio_features.extend(mspec[int(3 * num_mspec / 10):int(num_mspec * 7.5 / 10)])
        else :
            lang_audio_features.extend(mspec)
        
    all_language_features[path.rsplit('/',1)[1]] = lang_audio_features

print np.shape(all_language_features['odia'])
print np.shape(all_language_features['kannada'])
print np.shape(all_language_features['telugu'])
print np.shape(all_language_features['bengali'])


# In[101]:

from sklearn.mixture import GaussianMixture

all_language_gmm = {}

language_names = []

for key in all_language_features:
    all_language_gmm[key] = GaussianMixture(covariance_type='full' , n_components= 25).fit(all_language_features[key])
    language_names.append(key)
    print "fitting "+key


# In[93]:

print language_names


# In[104]:

from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# testing...

path_to_test_audio_file = "/home/anjani/AMP/assignment_3/lang_rec_data_with_division/test"

total = {}
correct = {}
incorrect = {}

for lang in language_names:
    total[lang] = 0
    correct[lang] = 0
    incorrect[lang] = 0

for lang in language_names:
    
    audio_features = []
    all_audio_files = glob.glob(path_to_test_audio_file+"/"+lang+"/*.wav") 
    max_match = 0 
    actual_label = lang
    
    for i in all_audio_files:
        
        sample_rate, X = scipy.io.wavfile.read(i)
        ceps, mspec, spec = mfcc(X)
        num_mspec = len(mspec)
        if lang == 'odia' :
            audio_features.extend(mspec[int(3 * num_mspec / 10):int(num_mspec * 7.5 / 10)])
        else :
            audio_features.extend(mspec)
        lang_arry = []
        predict_lable = ""
        
        for one_row in audio_features :
            max_match = 0
            predicted_lable = ""
            for each_lang in all_language_gmm :
                score = np.exp(all_language_gmm[each_lang].score_samples(one_row))
                if score >= max_match:
                    max_match = score
                    predicted_lable = each_lang

            lang_arry.append(predicted_lable)
        predict_lable = max(set(lang_arry), key = lang_arry.count)
    
        if actual_label == predict_lable:
            correct[actual_label] = correct[actual_label]+1
            # print actual_label, predict_lable, "..... correct", correct[actual_label]
        else:
            incorrect[actual_label] = incorrect[actual_label] + 1
            # print actual_label, predict_lable, "..... incorrect", incorrect[actual_label]
            
        total[actual_label] += 1

print correct
print incorrect
print total


# In[115]:

tot_correct = 0
tot_incorrect = 0

for lang in language_names:
    tot_correct += correct[lang]
    tot_incorrect += incorrect[lang]
accuracy = (1.0 * tot_correct)/(tot_correct + tot_incorrect)

print 'accuracy is : ',accuracy

