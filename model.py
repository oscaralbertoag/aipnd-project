""" ------------- Imports -------------
"""
from collections import OrderedDict
from torchvision import models
from torch import nn
import uuid
import image_classifier_constants as icc
import torch

""" ------------- Functions -------------
"""


def determine_device(gpu_selected):
    """ Determines the device to be used based on requested device and available device

    :param gpu_selected: boolean value provided by the user
    :return: device to be used based on choice and availability
    """
    if gpu_selected and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if gpu_selected: print("WARN: CUDA is not available. Using CPU instead")

    return device


def freeze_pretrained_model_params(pretrained_model):
    """ Freezes all pre-trained model params

        :param pretrained_model: a pre-trained model

        :return: same provided model with frozen parameters
    """
    for param in pretrained_model.parameters():
        param.requires_grad = False
    return pretrained_model


def fetch_feedforward_classifier_parameters(model, architecture):
    """ Fetches parameters for one of the 3 feed-forward classifiers created in this module
    :param model: contains the parameters to fetch
    :param architecture: name of the chosen architecture
    :return: model parameters
    """
    if architecture == icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50]:
        params = model.fc.parameters()
    else:
        params = model.classifier.parameters()

    return params


def build_feedforward_classifier(input_units, hidden_layer_inputs, output_units, dropout):
    """ Builds a feed-forward classifier based on the provided arguments. The output uses 'LogSoftmax'

        :param input_units: number of input units received by this classifier
        :param hidden_layer_inputs: list containing input units for each hidden layer to be added
        :param output_units: number of output units for this classifier
        :param dropout: dropout value to be used for each applicable layer

        :return: A feed-forward classifier
    """

    all_layer_inputs = [input_units] + hidden_layer_inputs

    classifier_elements = OrderedDict()

    # Build all layers, activation functions, and dropout
    for i, input_output_pair in enumerate(zip(all_layer_inputs, all_layer_inputs[1:])):
        classifier_elements['fc' + str(i + 1)] = nn.Linear(input_output_pair[0], input_output_pair[1])
        classifier_elements['relu' + str(i + 1)] = nn.ReLU()
        classifier_elements['dropout' + str(i + 1)] = nn.Dropout(dropout)

    # Build last layer
    classifier_elements['fc_last'] = nn.Linear(all_layer_inputs[-1], output_units)
    classifier_elements['output'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(classifier_elements)
    return classifier


def build_model(architecture, classifier_hidden_units_list, classifier_outputs, dropout):
    """ Loads a pre-trained model based on the provided architecture, and replaces the model classifier with a
        feed-forward classifier based on the provided hidden layers input units and number of outputs

        :param architecture: pre-trained model architecture to use. Valid values are 'alexnet', 'vgg19-bn', and 'resnet50'
        :param classifier_hidden_units_list: list of hidden layer inputs to use. e.g. [100, 50, 25] would use 3 hidden layers
                                      with 100, 50, and 25 input units each
        :param classifier_outputs: number of outputs for this model's feed forward classifier
        :param dropout: dropout value to be used to all layers except the last one

        :return: a pre-trained model (frozen params) with a feed-forward classifier defined by the provided arguments
    """

    architecture = architecture.lower()
    if architecture not in icc.SUPPORTED_ARCHITECTURES:
        raise Exception(
            f"Provided architecture '{architecture}' is not supported. Supported architectures: {icc.SUPPORTED_ARCHITECTURES}")

    model = None

    if architecture == icc.SUPPORTED_ARCHITECTURES[icc.ALEXNET]:
        model = models.alexnet(pretrained=True)
        freeze_pretrained_model_params(model)
        inputs = model.classifier[1].in_features
        model.classifier = build_feedforward_classifier(inputs, classifier_hidden_units_list, classifier_outputs,
                                                        dropout)
    elif architecture == icc.SUPPORTED_ARCHITECTURES[icc.VGG19_BN]:
        model = models.vgg19_bn(pretrained=True)
        freeze_pretrained_model_params(model)
        inputs = model.classifier[0].in_features
        model.classifier = build_feedforward_classifier(inputs, classifier_hidden_units_list, classifier_outputs,
                                                        dropout)
    elif architecture == icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50]:
        model = models.resnet50(pretrained=True)
        freeze_pretrained_model_params(model)
        inputs = model.fc.in_features
        model.fc = build_feedforward_classifier(inputs, classifier_hidden_units_list, classifier_outputs, dropout)

    return model


def save_checkpoint(model, class_to_index, optimizer, epochs, outputs, hidden_layers, dropout, architecture,
                    checkpoint_dir):
    """ Saves all important information about a model along with the checkpoint data
    :param model: model for which checkpoint is being created
    :param class_to_index: map containing class values to indexes
    :param optimizer: optimizer used along with current model
    :param epochs: number of epochs used to train the model
    :param outputs: number of output units
    :param hidden_layers: list of hidden layer inputs
    :param dropout: used during training for all classifier layers (except output layer)
    :param architecture: supported architecture used by this model
    :param checkpoint_dir: location where this checkpoint will be stored
    """
    inputs = None
    if architecture == icc.SUPPORTED_ARCHITECTURES[icc.ALEXNET]:
        inputs = model.classifier.fc1.in_features
    elif architecture == icc.SUPPORTED_ARCHITECTURES[icc.VGG19_BN]:
        inputs = model.classifier.fc1.in_features
    elif architecture == icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50]:
        inputs = model.fc.fc1.in_features

    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_index': class_to_index,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs,
                  'inputs': inputs,
                  'outputs': outputs,
                  'hidden_layers': hidden_layers,
                  'dropout': dropout,
                  'architecture': architecture}

    checkpoint_name = f"{architecture}-{uuid.uuid1()}.pth"
    full_path = f"{checkpoint_dir}/{checkpoint_name}"

    torch.save(checkpoint, full_path)
    print(f"Saved checkpoint at {full_path}")


def load_checkpoint(file_path, device):
    """ Loads a checkpoint onto a new model based on the checkpoint's configuration

        :param file_path: path containing the checkpoint file
        :param device: string that determines where to load the model/checkpoint: 'cuda' vs 'cpu'

        :return:  (model, checkpoint) pre-trained model with a state derived from the provided checkpoint,
        and the checkpoint used to load the model in case the caller needs additional stored information
    """

    device = device.lower()
    if device not in icc.DEVICES:
        raise Exception(f"Invalid device '{device}'! Valid options are: {icc.DEVICES}")

    if device == 'cpu':
        checkpoint = torch.load(file_path, map_location=device)
        pretrained_model = build_model(checkpoint['architecture'],
                                       checkpoint['hidden_layers'],
                                       checkpoint['outputs'],
                                       checkpoint['dropout'])
        pretrained_model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(file_path)
        pretrained_model = build_model(checkpoint['architecture'],
                                       checkpoint['hidden_layers'],
                                       checkpoint['outputs'],
                                       checkpoint['dropout'])
        pretrained_model.load_state_dict(checkpoint['state_dict'])
        pretrained_model = pretrained_model.to(torch.device(device))

    return pretrained_model, checkpoint
