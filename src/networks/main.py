from .mnist_LeNet import MNIST_LeNet_Encoder, MNIST_LeNet_Decoder, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet_Encoder, CIFAR10_LeNet_Decoder, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU_Encoder, CIFAR10_LeNet_ELU_Decoder, CIFAR10_LeNet_ELU_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    assert net_name in implemented_networks

    en_net = None
    de_net = None

    if net_name == 'mnist_LeNet':
        en_net = MNIST_LeNet_Encoder()
        de_net = MNIST_LeNet_Decoder()

    if net_name == 'cifar10_LeNet':
        en_net = CIFAR10_LeNet_Encoder()
        de_net = CIFAR10_LeNet_Decoder()

    if net_name == 'cifar10_LeNet_ELU':
        en_net = CIFAR10_LeNet_ELU_Encoder()
        de_net = CIFAR_10_LeNet_ELU_Decoder()

    return en_net, de_net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    return ae_net
