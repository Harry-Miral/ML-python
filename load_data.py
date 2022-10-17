import scipy.io as sciio
import numpy as np
import os


def Load(k):
    data = sciio.loadmat('./original_features/{}th_original_img_features.mat'.format(k))
    img_features = data['img_feature']
    labels = data['class']
    img_features_new = []
    for img in img_features:
        img_features_new.append(img.reshape(1, 48, 48))
    img_features_new = np.array(img_features_new)
    return img_features_new, labels
