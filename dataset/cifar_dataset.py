#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

from torchvision.datasets import CIFAR100
import numpy as np


class CIFAR100WithIdx(CIFAR100):
    """
    Extends CIFAR100 dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 rand_fraction=0.0):
        super(CIFAR100WithIdx, self).__init__(root=root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)

        assert (rand_fraction <= 1.0) and (rand_fraction >= 0.0)
        self.rand_fraction = rand_fraction

        if self.rand_fraction > 0.0:
            self.data = self.corrupt_fraction_of_data()

    def corrupt_fraction_of_data(self):
        """Corrupts fraction of train data by permuting image-label pairs."""

        # Check if we are not corrupting test data
        assert self.train is True, 'We should not corrupt test data.'

        nr_points = len(self.data)
        nr_corrupt_instances = int(np.floor(nr_points * self.rand_fraction))
        print('Randomizing {} fraction of data == {} / {}'.format(self.rand_fraction,
                                                                  nr_corrupt_instances,
                                                                  nr_points))
        # We will corrupt the top fraction data points
        corrupt_data = self.data[:nr_corrupt_instances, :, :, :]
        clean_data = self.data[nr_corrupt_instances:, :, :, :]

        # Corrupting data
        rand_idx = np.random.permutation(np.arange(len(corrupt_data)))
        corrupt_data = corrupt_data[rand_idx, :, :, :]

        # Adding corrupt and clean data back together
        return np.vstack((corrupt_data, clean_data))

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target, index



