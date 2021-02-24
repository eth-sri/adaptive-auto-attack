from zoo.mnist_model_pool import *
from utils.loader import get_loaders
from attack.attack_pool import *
from attack.network_transformation import *
import eagerpy as ep

device = 'cuda'
eps = 0.3
norm = ep.inf
ntest = 1000
# dataset
bounds = (0, 1)
trainN, train_loader, test_loader, input_size, input_channels, n_class = get_loaders('mnist', 128, 128)

# meta attack scheme
untargeted_attack = LinfProjectedGradientDescentAttack()
# attack = TargetedAttack(untargeted_attack)
attack = untargeted_attack

model_class_list = [Model1, Model2, Model3, Model4, Model5, Model6, Model7]
# model_class_list = [Model6, Model7]

for model_class in model_class_list:
    model_test = model_class(load=False, device=device)
    if not model_test.exists():
        model_test.train(train_loader, eps, save=True)
        auto_bpda_substitute(model_test.instance.model, test_loader)
        model_test.eval(test_loader, eps, attack, num=ntest, norm=norm)
    else:
        model_test = model_class(load=True, device=device)
        # print(defense_property_test(model_test.instance.model, test_loader))
        auto_bpda_substitute(model_test.instance.model, test_loader)
        model_test.eval(test_loader, eps, attack, num=ntest, norm=norm)

