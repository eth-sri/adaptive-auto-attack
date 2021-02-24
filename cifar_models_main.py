from attack.attack_base import TargetedAttack
from zoo.cifar_model_pool import *
from utils.loader import get_loaders
from attack.network_transformation import *
from attack.attack_pool import *
import eagerpy as ep

device = 'cuda'
eps = 4/255
norm = ep.inf
ntest = 1000
# dataset
bounds = (0, 1)
trainN, train_loader, test_loader, input_size, input_channels, n_class = get_loaders('cifar10', 128, 128)
x_test, y_test = get_data_tensor(test_loader, ntest, device=device)
# meta attack scheme

# attack = meta_attack.baseline_attack
# attack = LinfProjectedGradientDescentAttack()
# attack.params["EOT"] = 1
untargeted_attack = LinfProjectedGradientDescentAttack()
attack = TargetedAttack(untargeted_attack)

model_class_list = [Model7]

for model_class in model_class_list:
    model_test = model_class(load=False, device=device)
    if not model_test.exists():
        model_test.train(train_loader, eps, save=True)
        custom_bpda_substitute(model_test.instance.model, test_loader)
        model_test.eval(test_loader, eps, attack, num=ntest, norm=norm)
    else:
        model_test = model_class(load=True, device=device)
        auto_bpda_substitute(model_test.instance.model, test_loader)
        model_test.eval(test_loader, eps, attack, num=ntest, norm=norm)


