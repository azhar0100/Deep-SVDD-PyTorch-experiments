from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder, MNIST_LeNet_Decoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .toy_network import ToyNet, ToyNetAutoEncoder, ToyNetDecoder
import logging

logger = logging.getLogger("build_autoencoder.py")

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU','toynet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'toynet':
        net = ToyNet()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU','toynet')
    assert net_name in implemented_networks
    ae_net = None
    logger.info(net_name)

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'toynet':
        ae_net = ToyNetAutoEncoder()

    return ae_net


def build_decoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet','toynet')
    assert net_name in implemented_networks
    ae_net = None
    logger.info(net_name)

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Decoder()

    # if net_name == 'cifar10_LeNet':
    #     ae_net = CIFAR10_LeNet_Autoencoder()

    # if net_name == 'cifar10_LeNet_ELU':
    #     ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'toynet':
        ae_net = ToyNetDecoder()

    return ae_net
