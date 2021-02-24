import torch
from .resnet_block import ResNetBlock
from .wide_resnet_block import WideResNetBlock
import common.torch


class Classifier(torch.nn.Module):
    """
    Base classifier.
    """

    def __init__(self, N_class, resolution, **kwargs):
        """
        Initialize classifier.

        The keyword arguments, resolution, number of classes and other architecture parameters
        from subclasses are saved as attributes. This allows to easily save and load the model
        using common.state without knowing the exact architecture in advance.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution
        :type resolution: [int]
        """

        super(Classifier, self).__init__()

        assert N_class > 0, 'positive N_class expected'
        assert len(resolution) <= 3

        self.N_class = int(N_class)  # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        """ (int) Number of classes. """

        self.resolution = list(resolution)
        """ ([int]) Resolution as (channels, height, width) """

        self.kwargs = kwargs
        """ (dict) Kwargs. """

        self.include_clamp = self.kwargs_get('clamp', True)
        """ (bool) Whether to apply input clamping. """

        self.include_whiten = self.kwargs_get('whiten', False)
        """ (bool) Whether to apply input whitening/normalization. """

        self.include_scale = self.kwargs_get('scale', False)
        """ (bool) Whether to apply input scaling. """

        # __ attributes are private, which is important for the State to work properly.
        self.__layers = []
        """ ([str]) Will hold layer names. """

        self._N_output = self.N_class if self.N_class > 2 else 1
        """ (int) Number of outputs. """

        if self.include_clamp:
            self.append_layer('clamp', common.torch.Clamp())

        assert not (self.include_whiten and self.include_scale)

        if self.include_whiten:
            # Note that the weight and bias needs to set manually corresponding to mean and std!
            whiten = common.torch.Normalize(resolution[0])
            self.append_layer('whiten', whiten)

        if self.include_scale:
            # Note that the weight and bias needs to set manually!
            scale = common.torch.Scale(1)
            scale.weight.data[0] = -1
            scale.bias.data[0] = 1
            self.append_layer('scale', scale)

    def kwargs_get(self, key, default):
        """
        Get argument if not None.

        :param key: key
        :type key: str
        :param default: default value
        :type default: mixed
        :return: value
        :rtype: mixed
        """

        value = self.kwargs.get(key, default)
        if value is None:
            value = default
        return value

    def append_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.append(name)

    def prepend_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        self.insert_layer(0, name, layer)

    def insert_layer(self, index, name, layer):
        """
        Add a layer.

        :param index: index
        :type index: int
        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.insert(index, name)

    def forward(self, image, return_features=False):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :param return_features: whether to also return representation layer
        :type return_features: bool
        :return: logits
        :rtype: torch.autograd.Variable
        """

        features = []
        # print(image.shape)
        # image = torch.transpose(image, 1, 3)
        # print(image.shape)
        output = image

        # separate loops for memory constraints
        if return_features:
            for name in self.__layers:
                output = getattr(self, name)(output)
                features.append(output)
            return output, features
        else:
            for name in self.__layers:
                output = getattr(self, name)(output)
            return output

    def layers(self):
        """
        Get layer names.

        :return: layer names
        :rtype: [str]
        """

        return self.__layers

    def __str__(self):
        """
        Print network.
        """

        string = ''
        for name in self.__layers:
            string += '(' + name + ', ' + getattr(self, name).__class__.__name__ + ')\n'
            if isinstance(getattr(self, name), torch.nn.Sequential) or isinstance(getattr(self, name), ResNetBlock) or isinstance(getattr(self, name), WideResNetBlock):
                for module in getattr(self, name).modules():
                    string += '\t(' + module.__class__.__name__ + ')\n'
        return string

