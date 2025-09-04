#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta


# In[84]:


with open("../Data_Theodolite/TheoGelb_20250903_182754.td4") as f:
    data =f.read()

print(data[:])


# In[88]:


#with open("../Data_Theodolite/TheoRot_20250827_150544.txt", "r") as f:
    #content = f.read()

#print(content)


# In[89]:


theo_file = "../Data_Theodolite/TheoGelb_20250903_182754.td4"

azimuth_offset= 0

start_time = dt.datetime(2025, 9, 3, 8, 48, 9)

with open(theo_file, "r") as f:
    lines = f.readlines()
lines = lines[:-3]  # letzte drei Zeilen verwerfen

time_sec, value1, value2 = [], [], []
for line in lines:
    line = line.strip()
    if line.startswith("D"):
        parts = line.split()
        time_sec.append(float(parts[1]))
        value1.append(float(parts[2]))
        value2.append(float(parts[3]))
    elif line.startswith("S"):
        print("Metadata:", line)

azimuth = np.array(value1) + azimuth_offset
azimuth[azimuth > 360] -= 360

df_theo = pd.DataFrame({
    "time_sec": [start_time + timedelta(seconds=s) for s in time_sec],
    "azimuth": azimuth,
    "elevation":value2
})


# In[90]:


azimuth_mean=np.mean(df_theo["azimuth"])


# In[91]:


tower1=229.03
#right tower = light house


# In[92]:


tower2=180.83
#left tower = station


# In[93]:


theo_tower1= 289.53


# In[94]:


theo_tower2= 239.51


# In[95]:


difference1=theo_tower1-tower1
difference1


# In[96]:


difference2=theo_tower2-tower2
difference2


# In[97]:


mean_difference=(difference1+difference2)/2
mean_difference


# In[98]:


realaz = df_theo["azimuth"] - mean_difference


# In[62]:


realaz


# In[63]:


df_theo["realaz"] = realaz


# In[64]:


df_theo


# In[67]:


df_theo.to_csv("../Data_Theodolite/TheoRot_20250827_150544_corrected.csv", index=False)


# In[68]:


df_loaded = pd.read_csv("../Data_Theodolite/TheoRot_20250827_150544_corrected.csv")

print(df_loaded)


# In[ ]:





# In[ ]:




