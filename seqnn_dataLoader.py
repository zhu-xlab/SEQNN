import numpy as np
import random
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import os
import matplotlib.image as mpimg


def get_sat_data(path):
    data = scipy.io.loadmat(path)

    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    annotations = data['annotations']

    def reshape(train_x, train_y):
        out_x = np.zeros((train_x.shape[3], train_x.shape[0], train_x.shape[1], train_x.shape[2]))
        out_y = np.zeros((train_y.shape[1], train_y.shape[0]))
        for i in range(train_x.shape[3]):
            out_x[i,:,:,:] = train_x[:,:,:,i]
        for i in range(train_y.shape[1]):
            out_y[i,:] = train_y[:,i]
        return out_x, out_y
    def relabel(train_y, annotations):
        labels = {}
        for annotation in annotations:
            labels[annotation[0][0]]=annotation[1][0]
        output = []
        for i in range(train_y.shape[0]):
            temp = ''.join([str(int(x)) for x in train_y[i]])
            output.append(labels[temp])
        return np.array(output)
        
    def samples(train_x, train_y, n, labels):
        out_x, out_y = [],[]
        for label in labels:
            temp_x, temp_y = [], []
            for i in range(train_y.shape[0]):
                if label == train_y[i]:
                    temp_x.append(train_x[i])
                    temp_y.append(label)
            temp_x, temp_y = shuffle(np.array(temp_x), np.array(temp_y), random_state=33)
            out_x = out_x + list(temp_x[:n])
            out_y = out_y + list(temp_y[:n])
        return shuffle(np.array(out_x), np.array(out_y), random_state=33)

    def normalize(img):
        img = img.astype('float64')
        return (img - np.min(img))/(np.max(img)-np.min(img))

    def iqr(image):
        for i in range(image.shape[2]):
            boundry1, boundry2 = np.percentile(image[:,:,i], [2 ,98])
            image[:,:,i] = np.clip(image[:,:,i], boundry1, boundry2)
        return image
    
    def data_process(imgs, labels):
        labels = np.array([str(label).strip() for label in labels])
        processed_img = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = normalize(img)
            img = iqr(img)
            img = np.stack([np.pad(img[:, :, c], [(2, 2), (2, 2)], mode='constant') for c in range(4)], axis=2)
            processed_img.append(img)
        return np.array(processed_img), labels    
    
    train_x, train_y = reshape(train_x, train_y)
    test_x, test_y = reshape(test_x, test_y)
    train_y = relabel(train_y, annotations)
    test_y = relabel(test_y, annotations)
    labels = np.unique(train_y, return_counts=False)
    train_x, train_y = samples(train_x, train_y, 900, labels)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=1200, random_state=33)
    test_x, test_y = samples(test_x, test_y, 200, labels)
    
    train_x, train_y = data_process(train_x, train_y)
    valid_x, valid_y = data_process(valid_x, valid_y)
    test_x, test_y = data_process(test_x, test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_lcz_data(path):
    rawdata = scipy.io.loadmat(path)
    data = rawdata['setting0']
    train_x = data['train_x'][0][0]
    train_y = data['train_y'][0][0][0]
    test_x = data['test_x'][0][0]
    test_y = data['test_y'][0][0][0]
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=2000, random_state=42)

    def iqr(image):
        for i in range(image.shape[2]):
            boundry1, boundry2 = np.percentile(image[:,:,i], [2 ,98])
            image[:,:,i] = np.clip(image[:,:,i], boundry1, boundry2)
        return image

    def normalize(img):
        img = img.astype('float64')
        return (img - np.min(img))/(np.max(img)-np.min(img))
    
    def data_process(imgs, labels):
        labels = np.array([str(label).strip() for label in labels])
        processed_img = []
        for i in range(imgs.shape[0]):
            img, _ = imgs[i]
            img = iqr(img)
            img = normalize(img)
            processed_img.append(img)
        return np.array(processed_img), labels
    
    train_x, train_y = data_process(train_x, train_y)
    valid_x, valid_y = data_process(valid_x, valid_y)
    test_x, test_y = data_process(test_x, test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_overhead_data(path):
    def normalize(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    def iqr(image):
        boundry1, boundry2 = np.percentile(image, [2 ,98])
        image = np.clip(image, boundry1, boundry2)
        return image
    
    def load_images(folder, label):
        images = []
        for filename in os.listdir(folder):
            img = mpimg.imread(os.path.join(folder, filename))
            img = normalize(img)
            img = iqr(img)
            img = np.pad(img, [(2, 2), (2, 2)], mode='constant')
            img = img.reshape(32, 32, 1)
            images.append(img)
        return images, [label] * len(images)
    
    train_x, train_y, test_x, test_y = [], [], [], []
    training_path = os.path.join(path, 'training')
    test_path = os.path.join(path, 'testing')
    labels = ['car', 'ship', 'plane', 'harbor', 'parking_lot']
    for label in labels:
        temp_training_path = os.path.join(training_path, label, '')
        temp_test_path = os.path.join(test_path, label, '')
        temp_x, temp_y = load_images(temp_training_path, label)
        train_x = train_x + temp_x
        train_y = train_y + temp_y
        temp_x, temp_y = load_images(temp_test_path, label)
        test_x = test_x + temp_x
        test_y = test_y + temp_y
        
    train_x, train_y = shuffle(np.array(train_x), np.array(train_y), random_state=33)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.15, random_state=33)
    test_x, test_y = shuffle(np.array(test_x), np.array(test_y), random_state=33)
    return train_x, train_y, valid_x, valid_y, test_x, test_y
    

class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        
        if dataset == 'sat':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_sat_data('Data/SAT-6/sat-6-full.mat')
        if dataset == 'lcz':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_lcz_data('Data/LCZ/data_5fold_5classes.mat')
        if dataset == 'overhead':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_overhead_data('Data/overhead')

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y   
        
    def get_categories(self):
        class_name = [str(x).strip() for x in np.unique(self.train_y)]
        return class_name
    
    def get_data(self): 
        train_y = LabelBinarizer().fit_transform(self.train_y)
        valid_y = LabelBinarizer().fit_transform(self.valid_y)
        test_y = LabelBinarizer().fit_transform(self.test_y)
        return self.train_x, train_y, self.valid_x, valid_y, self.test_x, test_y
    


    
