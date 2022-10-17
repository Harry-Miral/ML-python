import random
import scipy.io
from tqdm import tqdm
import os
from PIL import Image
import torch
import numpy as np
import cv2


def draw_CAM(model, img_path, save_path, device):
    shape = 48
    img = Image.open(img_path)
    img_ = img.resize((shape, shape))
    array = np.asarray(img_, dtype="float32") / 255
    input_ = torch.tensor(array.reshape(1, shape, shape)).view((1, 1, shape, shape)).to(device)
    model.eval()
    features = model.feature(input_)
    output = model.classifier(features.view(-1, 6 * 6 * 32))

    def extract(g):
        global features_grad
        features_grad = g

    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()

    grads = features_grad

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(32):
        features[i, ...] *= pooled_grads[i, ...]
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(save_path, superimposed_img)


def Generate(k, new_index):
    savepath = r'features_whitegirl'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('./model_dict/{}th_Net.pkl'.format(k))
    model.eval()
    model.to(device)
    index = new_index
    shape = 48
    train_size = int(len(index) * 0.9)
    train_index = index[:train_size]
    test_index = index[train_size:]

    train_path = r'./features_whitegirl/train'
    test_path = r'./features_whitegirl/test'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    # "{}/test/{}th_model/{}.png"

    if not os.path.exists(train_path + '/{}th_model'.format(k)):
        os.mkdir(train_path + '/{}th_model'.format(k))
    if not os.path.exists(test_path + '/{}th_model'.format(k)):
        os.mkdir(test_path + '/{}th_model'.format(k))

    print('Generate {}th model train heat map...'.format(k))
    for i in tqdm(range(len(train_index))):
        img_path = './figs/fig.' + str(train_index[i]) + '.jpg'
        save_path = "{}/train/{}th_model/{}.png".format(savepath, k, train_index[i])
        draw_CAM(model, img_path, save_path, device)

    print('Generate {}th model test heat map...'.format(k))
    #
    for i in tqdm(range(len(test_index))):
        img_path = './figs/fig.' + str(test_index[i]) + '.jpg'
        save_path = "{}/test/{}th_model/{}.png".format(savepath, k, test_index[i])
        draw_CAM(model, img_path, save_path, device)
    # 
    print('Generate {}th img feature...'.format(k))
    features = []
    for i in tqdm(range(len(index))):
        img = Image.open('./figs/fig.' + str(index[i]) + '.jpg')
        img_ = img.resize((shape, shape))
        array = np.asarray(img_, dtype="float32") / 255
        input_ = torch.tensor(array.reshape(1, shape, shape)).view((1, 1, shape, shape)).to(device)
        with torch.no_grad():
            _, feature = model(input_)
            features.append(feature.cpu().numpy()[0])
    features = np.array(features)
    scipy.io.savemat('./new_features/{}th_new_img_features.mat'.format(k), {'hidden_features': features})
