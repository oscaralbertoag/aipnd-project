""" ------------- Imports -------------
"""
import argparse
import torch
import image_utils as iu
import model as m
import json

""" ------------- Functions -------------
"""


def get_predict_arguments():
    """ Fetches all necessary inputs from the user

        Returns
        -------
        Namespace object containing all key-value pairs for user input arguments
    """

    parser = argparse.ArgumentParser(description='Uses a pre-trained model to make a prediction on an image')

    parser.add_argument('image_path', type=str, help="file path to the target image")
    parser.add_argument('checkpoint_path', type=str, help='file path to the model checkpoint to use for predictions')
    parser.add_argument('--top_k', type=int, default=5,
                        help="determines the number of top classes to return; defaults to 5")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='JSON file containing mappings to be used (categories-to-names)')
    parser.add_argument('--gpu', default=False, nargs='?', const=True,
                        help='if present, GPU will be used for inference; defaults to CPU')

    return parser.parse_args()


def predict(image_path, model, device, top_k=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.

        Arguments
        ---------
        :param image_path: path to the image to load and run through our model prediction
        :param model: pre-trained model to use for flower category prediction
        :param top_k: number of top identified classes to return
        :param device: determines where prediction will be executed; values are 'cuda' or 'cpu'

        Returns
        -------
        top_probabilities: top K probabilities predicted by the model
        top_classes: top K classes predicted by the model
    """

    processed_img = torch.from_numpy(iu.pre_process_image(image_path)).float()
    processed_img = torch.unsqueeze(processed_img, 0)
    processed_img = processed_img.to(torch.device(device))

    model.eval()

    with torch.no_grad():
        logps = model(processed_img)
        ps = torch.exp(logps)
        top_probabilities, top_classes = ps.topk(top_k, dim=1)

    return top_probabilities, top_classes


""" ------------- Main -------------
"""


def main():
    # Fetch user input
    user_input = get_predict_arguments()
    image_path = user_input.image_path
    checkpoint_path = user_input.checkpoint_path
    top_k = user_input.top_k
    category_names_file_path = user_input.category_names
    gpu = user_input.gpu
    device = m.determine_device(gpu)

    # Load model and checkpoint
    model, checkpoint = m.load_checkpoint(checkpoint_path, device)

    # Predict
    top_probabilities, top_classes = predict(image_path, model, device, top_k)

    top_classes.squeeze_()
    top_probabilities.squeeze_()

    # Load categories-to-names
    with open(category_names_file_path, 'r') as f:
        cat_to_name = json.load(f)

    classes_to_indexes = checkpoint['class_to_index']
    indexes_to_classes = {v: k for k, v in classes_to_indexes.items()}

    named_classes = [cat_to_name[indexes_to_classes[top_class]] for top_class in top_classes.cpu().numpy()]

    print("Prediction Results")
    print(f"- Model Used: {checkpoint['architecture']}")
    print("  - Details:")
    print(f"    - Inputs: {checkpoint['inputs']}\n", f"    - Outputs: {checkpoint['outputs']}\n",
          f"    - Hidden Layers: {checkpoint['hidden_layers']}\n", f"    - Dropout: {checkpoint['dropout']}\n"
                                                                   f"    - Epochs: {checkpoint['epochs']}\n")

    print(f"Top {top_k} Probabilities:")
    for named_class, probability in zip(named_classes, top_probabilities):
        print("- Predicted: {} --> {:.3f}".format(named_class.capitalize(), probability))


if __name__ == '__main__':
    main()
