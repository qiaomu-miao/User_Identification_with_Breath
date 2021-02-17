#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import os
import math
import pickle


# In[31]:


base_path = "/home/qiaomu/HCI_project/project"
feature_path = os.path.join(base_path, "features")
filenames = os.listdir(feature_path)
time_recording_path = os.path.join(base_path, "time_recording.xlsx")


# In[32]:


times_df = pd.read_excel(time_recording_path, index_col=0)


# In[33]:


times_df = pd.read_excel(time_recording_path, index_col=0)
breath_info, final_data = {}, {}
for index, row in times_df.iterrows():
    max_col = np.amax(np.argwhere(np.invert(pd.isna(row).values))) + 1
    print(index)
    print(row.values)
    print(max_col)
    person = index.strip()
    breath_info[person] = dict()
    breath_info[person]["breath_cycles"] = int(max_col / 2)
    breath_info[person]['time_info'] = [tuple(row.values[i:i+2]) for i in range(0,max_col,2)]


# In[34]:


with open(os.path.join(base_path, "breath_labels_new.pickle"), "wb") as file:
    pickle.dump(breath_info, file, protocol = pickle.HIGHEST_PROTOCOL)


# In[35]:


for person in breath_info.keys():
    feature_file = os.path.join(base_path, "features", person+".csv")
    df = pd.read_csv(feature_file)
    data = df.values
    max_frame = data.shape[0]
    final_data[person] = list()
    for start_time, end_time in breath_info[person]['time_info']:
        frame_start, frame_end = max(math.floor((start_time - 0.015) / 0.01),0), min(math.ceil((end_time - 0.015) / 0.01), max_frame)
        data_append = data[frame_start:frame_end+1]
        final_data[person].append(data_append)
        print(data_append.shape)
    assert len( final_data[person]) == breath_info[person]["breath_cycles"], "registered data length and recorded cycle mismatch!"
    


# In[36]:


with open(os.path.join(base_path, "extracted_features_new.pickle"), "wb") as file:
    pickle.dump(final_data, file, protocol = pickle.HIGHEST_PROTOCOL)


# In[ ]:




