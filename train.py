""" ------------- Imports -------------
"""
import argparse
import datetime
import time
import image_classifier_constants as icc
import model as m
import load
import torch
from torch import nn, optim
import json

""" ------------- Functions -------------
"""


def get_train_arguments():
    """ Fetches all necessary inputs from the user

        Returns
        -------
        Namespace object containing all key-value pairs for user input arguments
    """

    parser = argparse.ArgumentParser(description='Trains an image classifier')

    parser.add_argument('data_directory', type=str, default='flowers',
                        help="directory containing all data to be used to train a model. Expects this directory to contain: 'train', 'valid', and 'test' directories")
    parser.add_argument('--save_dir', type=str, default='saved-checkpoints',
                        help='custom directory to save model checkpoints')
    parser.add_argument('--arch', type=str, default=icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50],
                        help=f"choice of model architecture to use; defaults to {icc.SUPPORTED_ARCHITECTURES[icc.RESNET_50]}",
                        choices=icc.SUPPORTED_ARCHITECTURES)
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='set the learning rate to be used; defaults to 0.0003')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[512],
                        help='list of hidden units to use; for example, if 1024 and 512 are provided, the resulting feed-forward classifier will have 2 hidden layers')
    parser.add_argument('--epochs', type=int, default=8,
                        help='number of epochs to train; defaults to 8')
    parser.add_argument('--gpu', default=False, nargs='?', const=True,
                        help='if present, GPU will be used to train the model; defaults to CPU')
    parser.add_argument('--print_every', type=int, default=100,
                        help='how many steps between printing model stats (e.g. accuracy, loss); a smaller number will slow down processing. Defaults to 100')
    parser.add_argument('--test', default=False, nargs='?', const=True,
                        help='if present, run model through testing datapoints after training; skips testing cycle by default')

    return parser.parse_args()


def determine_device(gpu_selected):
    device = None
    if gpu_selected and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if gpu_selected: print("WARN: CUDA is not available. Using CPU instead")

    return device


def validate(model, criterion, validation_dataloader, device):
    ''' Runs through validation data and calculates and returns loss and accuracy

        Arguments
        ---------
        model: model to validate
        criterion: used to calculate model validation loss
        validation_dataloader: dataloader for validation data
        device: 'cpu' or 'gpu'

        Returns
        -------
        accuracy: calculated accuracy for all validation data
        validation_loss: calculated loss for all validation data
    '''
    # Put model in 'evaluation' mode
    model.eval()

    with torch.no_grad():
        validation_loss = 0
        accuracy = 0
        for inputs, labels in validation_dataloader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            validation_logps = model.forward(inputs)
            # Calculate loss
            loss = criterion(validation_logps, labels)
            validation_loss += loss.item()
            # Calculate accuracy
            validation_ps = torch.exp(validation_logps)
            top_p, top_class = validation_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    # Put model back in 'training' mode
    model.train()

    return accuracy, validation_loss


def train(model, device, epochs, optimizer, criterion, training_dataloader, validation_dataloader, print_every=0):
    ''' Trains the provided model with the following arguments:

        Arguments
        ---------
        model: model to be used to train
        device: 'cpu' or 'gpu'
        epochs: number of training iterations over the entire learning dataset
        optimizer: optimizer to be used to update weights after backpropagation (already configured with a learn rate)
        criterion: used to calculate model loss and execute backpropagation
        training_dataloader: dataloader for training data
        validation_dataloader: dataloader for validation data
        print_every: will print intermediate step stats if provided value > 0. Defaults to zero (no intermediate printing)

        Returns
        -------
        train_losses: list of all training loss values that can be used in plots
        validation_losses: list of all validation loss values that can be used in plots
    '''
    print(
        f"\n****TRAINING (printing model stats every {print_every} steps)****\n Device: {device}, Epochs: {epochs}, Training batches: {len(training_dataloader)},  Validation batches: {len(validation_dataloader)}")

    model.train()
    step = 0
    training_loss = 0
    train_losses, validation_losses = [], []

    for epoch in range(epochs):

        for inputs, labels in training_dataloader:
            step += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # Reset the gradients for every pass
            optimizer.zero_grad()

            # 1. Forward pass
            logps = model.forward(inputs)
            # 2. Calculate loss
            loss = criterion(logps, labels)
            # 3. Backpropagate
            loss.backward()
            # 4. Update weights
            optimizer.step()

            training_loss += loss.item()

            # Run Validations
            if print_every > 0 and step % print_every == 0:
                validation_loss = 0
                accuracy = 0

                # Validate
                calculated_accuracy, calculated_validation_loss = validate(model, criterion, validation_dataloader,
                                                                           device)
                validation_loss += calculated_validation_loss
                accuracy += calculated_accuracy

                # Update loss lists
                train_losses.append(training_loss / print_every)
                validation_losses.append(validation_loss / len(validation_dataloader))

                print("Step {} - Epoch: {}/{} | ".format(step, epoch + 1, epochs),
                      "Training Loss: {:.3f} | ".format(training_loss / print_every),
                      "Validation Loss: {:.3f} | ".format(validation_loss / len(validation_dataloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validation_dataloader)))
                training_loss = 0

    return train_losses, validation_losses


def test(model, criterion, testing_dataloader, device):
    ''' Iterates through test dataset to verify the accuracy of the trained model

        Arguments
        ---------
        model: used for predictions on the test dataset
        criterion: used to calculate loss
        testing_dataloader: data loader containing testing dataset
        device: 'cpu' or 'gpu'
    '''

    model.eval()
    print(f"Testing model (batches: {len(testing_dataloader)})")

    with torch.no_grad():
        test_loss = 0
        test_accuracy = 0

        for images, labels in testing_dataloader:
            # Move images and labels to selected device
            images, labels = images.to(device), labels.to(device)

            test_logps = model(images)
            test_loss += criterion(test_logps, labels)

            # Calculate Accuracy
            test_ps = torch.exp(test_logps)
            top_p, top_class = test_ps.topk(1, dim=1)
            test_equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(test_equals.type(torch.FloatTensor))

        print("Test Loss: {:.3f}".format(test_loss / len(testing_dataloader)),
              " | Test Accuracy: {:.3f}".format(test_accuracy / len(testing_dataloader)))


""" ------------- Main -------------
"""


def main():
    # Fetch user arguments
    user_input = get_train_arguments()
    data_dir = user_input.data_directory
    checkpoint_save_dir = user_input.save_dir
    architecture = user_input.arch
    classifier_hidden_units_list = user_input.hidden_units
    dropout = 0.2
    epochs = user_input.epochs
    learn_rate = user_input.learning_rate
    print_every = user_input.print_every
    device = torch.device(determine_device(user_input.gpu))

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    classifier_outputs = len(cat_to_name)

    print("Configuring training session with the following parameters:\n",
          f"- Architecture: {architecture}\n", f"- Hidden layer inputs: {classifier_hidden_units_list}\n",
          f"- Classifier Outputs: {classifier_outputs}\n", f"- Dropout: {dropout}\n", f"- Epochs: {epochs}\n",
          f"- Learn Rate: {learn_rate}\n", f"- Device: {device}\n")

    # Load training and validation data
    training_dataloader, training_dataset = load.load_data(data_dir, icc.DIRECTORIES[icc.TRAIN_DIR])
    validation_dataloader, validation_dataset = load.load_data(data_dir, icc.DIRECTORIES[icc.VALID_DIR])

    # Load pre-trained model with new classifier
    model = m.build_model(architecture, classifier_hidden_units_list, classifier_outputs, dropout)
    model.to(device)

    # Create an optimizer to update the weights
    classifier_params = m.fetch_feedforward_classifier_parameters(model, architecture)
    optimizer = optim.Adam(classifier_params, lr=learn_rate)
    criterion = nn.NLLLoss()

    # Train
    print(f"\nStarting training: {datetime.datetime.now()}")
    start = time.time()

    train_losses, validation_losses = train(model, device, epochs, optimizer, criterion, training_dataloader,
                                            validation_dataloader, print_every)

    print(f"Finished training: {datetime.datetime.now()}. Total run time: {time.time() - start}")

    # Test (optional)
    if user_input.test:
        testing_dataloader, testing_dataset = load.load_data(data_dir, icc.DIRECTORIES[icc.TEST_DIR])

        test_start = time.time()
        test(model, criterion, testing_dataloader, device)

        print(f"Testing time: {time.time() - test_start}")

    # Save checkpoint
    m.save_checkpoint(model, training_dataset.class_to_idx, optimizer, epochs, classifier_outputs,
                      classifier_hidden_units_list, dropout, architecture, checkpoint_save_dir)


if __name__ == '__main__':
    main()
