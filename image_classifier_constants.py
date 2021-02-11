# Contains any values to be used across modules
SUPPORTED_ARCHITECTURES = ['resnet50', 'alexnet', 'vgg19-bn']
RESNET_50 = 0
ALEXNET = 1
VGG19_BN = 2

DEVICES = ['cpu', 'cuda']
CPU = 0
GPU = 1

DIRECTORIES = ['train', 'test', 'valid']
TRAIN_DIR = 0
TEST_DIR = 1
VALID_DIR = 2