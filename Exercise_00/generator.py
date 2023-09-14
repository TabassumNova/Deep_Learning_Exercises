import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
import random
import math

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        # TODO: implement constructor
        # path = "C:/NOVA/MSC@FAU/Deep Learning/Exercise/exercise0_material/exercise0_material/src_to_implement/data"
        # self.file_path = os.path.join(path, file_path)
        # self.label_path = os.path.join(path, label_path)
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size_x, self.image_size_y, self.image_size_c = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.batch_num = math.ceil(100 / self.batch_size)
        self.temp = 0

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        label_file = open(self.label_path)
        label = json.load(label_file)
        training_data = []
        for img in os.listdir(self.file_path):
            img_array = np.load(os.path.join(self.file_path, img))
            new_array = resize(img_array, (self.image_size_x, self.image_size_y))  # next()
            if self.mirroring or self.rotation:  # mirroring or rotate
                new_array = self.augment(new_array)

            training_data.append((new_array, label[img[:-4]]))

        if self.shuffle:      #shuffle
            random.shuffle(training_data)

        subset = math.ceil(100/self.batch_size)
        final = []
        x = 0
        for i in range (0,subset):
            if (x+self.batch_size)>100:
                extra = x+self.batch_size-100
                temp = self.batch_size-extra
                final.extend(training_data[x:x+temp])
                final.extend(training_data[0:extra])
            else:
                final.extend(training_data[x:x+self.batch_size])
            x = x+self.batch_size

        im = []
        la = []

        for i in range(0, len(final)):
            im.append(final[i][0])
            la.append(final[i][1])

        images = np.array(im[self.temp * self.batch_size: self.temp * self.batch_size + self.batch_size])
        labels = np.array(la[self.temp * self.batch_size: self.temp * self.batch_size + self.batch_size])
        self.temp += 1

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if self.mirroring:
            img = np.flip(img, axis=1)
        if self.rotation:
            angle = random.choice([0, 90, 180, 270])
            img = rotate(img, angle)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        images, labels = self.next()
        fig = plt.figure()
        x = math.ceil(self.batch_size/3)
        for i in range (0, self.batch_size):
            a = fig.add_subplot(x,3,i+1)
            plt.imshow(images[i])
            a.title.set_text(self.class_name(labels[i]))
        plt.show()