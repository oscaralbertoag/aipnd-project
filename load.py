""" ------------- Imports -------------
"""
import image_classifier_constants as icc
from torchvision import datasets, transforms
import torch

""" ------------- Functions -------------
"""


def create_validation_transform():
    """ Creates a transformation to be used for validations and testing
    :return: transformations (resize, center crop, normalization)
    """
    return transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])


def create_training_transform():
    """ Creates a transformation to be used for training
    :return:  transformations (random rotation, resizing, cropping, flipping, and normalization)
    """
    return transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])


def load_data(data_dir, data_type):
    """ Loads data from a specific directory to be used in training, validation, or testing
    :param data_dir: location of data to load
    :param data_type: type of data to load (train, valid, test)
    :return: (data_loader, dataset) loaded data in batches of 32
    """
    if data_type not in icc.DIRECTORIES:
        raise Exception(f"Invalid data type: {data_type}. Options: {icc.DIRECTORIES}")

    full_dir = data_dir + '/' + data_type
    transformation = create_training_transform() if data_type == 'train' else create_validation_transform()
    dataset = datasets.ImageFolder(full_dir, transform=transformation)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Loaded {len(dataset)} data points ({data_type}) split into {len(data_loader)} batches")
    return data_loader, dataset


