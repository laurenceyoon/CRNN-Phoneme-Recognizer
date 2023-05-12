# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from pyts.metrics import dtw, itakura_parallelogram, sakoe_chiba_band
import librosa
import librosa.display
import numpy as np
import scipy
import pandas as pd
import yaml
from attributedict.collections import AttributeDict
import matplotlib
import soundfile as sf
import re
from phonerec.models import *
from phonerec.utils import visualize_with_feature, load_model, load_yaml


# =================================== Common
# %%
with open("./timit61.phoneset") as f:
    x_axis_labels = f.read().splitlines()

# =================================== oiseaux 1
# %%
model_path = "./pt/oiseaux1_39.pt"
data = torch.load(model_path).squeeze()
fig, ax = plt.subplots(figsize=(50, 20))
plt.imshow(data.T, aspect="auto")

# =================================== oiseaux 2
# %%
model2_path = "./pt/oiseaux2_39.pt"
data2 = torch.load(model2_path).squeeze()
fig, ax = plt.subplots(figsize=(50, 20))
plt.imshow(data2.T, aspect="auto")
# np.argmax(data.T)

# =================================== DTW
# %%
# D, wp = librosa.sequence.dtw(data.T, data2.T, subseq=True)
D, wp = librosa.sequence.dtw(data.T, data2.T)

# =================================== figure
# %%
dist = scipy.spatial.distance.cdist(data, data2)  # [d, wy]
plt.figure(figsize=(20, 20))
plt.imshow(dist.T, aspect='auto', origin='lower', interpolation='nearest')

from matplotlib import cm
cmap = cm.get_cmap('magma', 100)
plt.plot(wp[:, 0], wp[:, 1], '.', color='r')

# x,y  = zip(*odtw.candi_history)

# from matplotlib import cm
# cmap = cm.get_cmap('magma', 100)
# for n in range(len(x)):
#     plt.plot(x[n], y[n], '.', color='r')

# %%
data1_label = np.argmax(data, axis=1)
data2_label = np.argmax(data2, axis=1)

# %%
fig, ax = plt.subplots(figsize=(50, 20))
ax_with_lbl = ax.twinx()
ax.imshow(data.T, aspect="auto", origin="lower")
ax_with_lbl.imshow(data1_label, aspect="auto", interpolation="none", origin="lower", alpha=0.5, vmin=0.49, vmax=0.5)

# %%
