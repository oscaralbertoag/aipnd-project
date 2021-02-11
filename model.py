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
    if gpu_selected and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if gpu_selected: print("WARN: CUDA is not available. Using CPU instead")

    return device


def freeze_pretrained_model_params(pretrained_model):
    """ Freezes all pre-trained model params

        Arguments
        ---------
        pretrained_model: a pre-trained model

        Returns
        -------
        pretrained_model: same provided model with frozen parameters

    """
    for param in pretrained_model.parameters():
        param.requires_grad = False
    return pretrained_model


def fetch_feedforward_classifier_parameters(model, architecture):
    params = None
    if architecture == icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50]:
        params = model.fc.parameters()
    else:
        params = model.classifier.parameters()

    return params


def build_feedforward_classifier(input_units, hidden_layer_inputs, output_units, dropout):
    """ Builds a feed-forward classifier based on the provided arguments. The output uses 'LogSoftmax'

        Arguments
        ---------
        input_units: number of input units received by this classifier
        hidden_layer_inputs: list containing input units for each hidden layer to be added
        output_units: number of output units for this classifier
        dropout: dropout value to be used for each applicable layer

        Returns
        -------
        A feed-forward classifier
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

        Arguments
        ---------
        architecture: pre-trained model architecture to use. Valid values are 'alexnet', 'vgg19-bn', and 'resnet50'
        classifier_hidden_units_list: list of hidden layer inputs to use. e.g. [100, 50, 25] would use 3 hidden layers
                                      with 100, 50, and 25 input units each
        classifier_outputs: number of outputs for this model's feed forward classifier
        dropout: dropout value to be used to all layers except the last one

        Returns
        -------
        A pre-trained model (frozen params) with a feed-forward classifier defined by the provided arguments
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
    ''' Loads a checkpoint onto a new model based on the checkpoint's configuration

        Arguments
        ---------
        file_path: path containing the checkpoint file
        device: string that determines where to load the model/checkpoint: 'cuda' vs 'cpu'

        Returns
        -------
        model: A pre-trained model with a state derived from the provided checkpoint
        checkpoint: The checkpoint used to load the model in case the caller needs additional stored information
    '''

    valid_devices = ['cpu', 'cuda']
    device = device.lower()
    if device not in valid_devices:
        raise Exception(f"Invalid device '{device}'! Valid options are: {valid_devices}")

    checkpoint = None
    pretrained_model = None

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


def main():
    vgg19 = models.vgg19_bn(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)
    print("ORIGINAL PRE-TRAINED")
    print(f"VGG19: \n {vgg19}\n")
    print(f"AlexNet: \n {alexnet}\n")
    print(f"ResNet50: \n {resnet50}\n")

    print("\nMODIFIED PRE-TRAINED (REPLACED CLASSIFIERS)\n")
    vgg19_a = build_model(icc.SUPPORTED_ARCHITECTURES[icc.VGG19_BN], [600, 400], 102, 0.2, True)
    alexNet_a = build_model(icc.SUPPORTED_ARCHITECTURES[icc.ALEXNET], [600, 400], 102, 0.2, True)
    resnet50_a = build_model(icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50], [600, 400], 102, 0.2, True)
    print(f"VGG19: \n {vgg19_a}\n")
    print(f"AlexNet: \n {alexNet_a}\n")
    print(f"ResNet50: \n {resnet50_a}\n")


if __name__ == '__main__':
    main()
