import scipy.io
import random
import numpy as np
from sklearn.preprocessing import StandardScaler

# restore data
PATH = "/home/yulia/pepper_ws/src/action_recognition/scripts/learning/database/"
data = scipy.io.loadmat(PATH + "original_normalized_data.mat")
x = 1*data['data']

# normalize data
copy1 = np.array([])
for line in x:
    y = np.append(line, line)
    copy1 = np.append(copy1, y)

# fake missing modalities
copy = np.array([])
j = np.array([-2, -2, -2, -2, -2, -2])
p = np.array([-2, -2, -2, -2])
for y in x:
    z = np.array([])
    b = random.choice([True, False])
    if b:
        z = np.append(j, y[6:])
    else:
        z = np.append(y[:6],p)
    y = np.append(y, z)
    copy = np.append(copy, y)
out = np.append(copy1, copy, axis=0)

# fake prediction modality
copy2 = np.array([])
j = np.array([-2, -2])
k= np.array([-2, -2, -2])
for y in x:
    z = np.concatenate((y[:3], k, y[6:8], j))
    y = np.append(y, z)
    copy2 = np.append(copy2, y)


out = np.append(out, copy2, axis=0)
print copy1.shape
print copy2.shape
print copy.shape
out = out.reshape(15000, 20)

#x = np.append(x,copy, axis=0)'''

scipy.io.savemat(PATH + 'augm_rec_pred_data.mat', {'data': out})

