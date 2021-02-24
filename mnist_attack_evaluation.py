from zoo.mnist_model_pool import *
from utils.loader import get_loaders
from attack.attack_pool import *
import eagerpy as ep
from utils.eval import *

# MNIST eps=0.3 test
device = 'cuda'
epsilon = 0.3
norm = ep.inf
# epsilon = 2
# norm = 2
ntest = 1000

# dataset
bounds = (0, 1)
trainN, train_loader, test_loader, input_size, input_channels, n_class = get_loaders('mnist', 128, 128)
x_test, y_test = get_data_tensor(test_loader, ntest, device=device)

# meta attack scheme
# attack = LinfDeepFoolAttack()
# attack = LinfBrendelBethgeAttack()

attack_list = [LinfFastGradientAttack(), LinfProjectedGradientDescentAttack(), LinfDeepFoolAttack()]
model_class_list = [Model1(load=True), Model2(load=True), Model3(load=True), Model4(load=True), Model5(load=True), Model6(load=True), Model7(load=True)]


result = get_result_from_loader(attack_list, model_class_list, test_loader, epsilon, ntest, norm)
print_robustness_time_disturbance_table(result)
