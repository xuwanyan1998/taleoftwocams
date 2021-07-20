import random
from skimage import io
from tensorflow import keras
import numpy as np
from Utils import pack_raw


class TwoCamDatasets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_size, input_img_paths, target_img_paths,packraw=True,randomfliping=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_size = input_img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.packraw = packraw
        self.randomfliping = randomfliping

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        xi,yi,wi,hi = random.randint(50,400),random.randint(50,400),self.input_img_size,self.input_img_size
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
#         print(batch_input_img_paths,batch_target_img_paths)
#         x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")

        if self.packraw:
            x = np.zeros((self.batch_size,) + (256, 256) + (4,), dtype="float32")
        else:
            x = np.zeros((self.batch_size,) + (512, 512) + (1,), dtype="float32")
        flip_flag = False

        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path)
            img = img[xi:xi+wi,yi:yi+hi]
#             img = (img / 127.5) - 1
            if self.packraw:
                x[j] = pack_raw(img)
            else:
                x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = io.imread(path)
            img = img[xi:xi+wi,yi:yi+hi]
#             img = (img / 127.5) - 1
            y[j] = np.expand_dims(img, 2)

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=2)
        # print(x.shape)
        # if np.random.randint(2, size=1)[0] == 1:  # random transpose
        #     x = np.transpose(x, (0, 2, 1))
        #     y = np.transpose(y, (0, 2, 1))

        return x, y

