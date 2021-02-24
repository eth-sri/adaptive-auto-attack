from zoo.cifar_model_pool import *
from attack.attack_pool import *
from attack.bpda_approx import *
import eagerpy as ep
from attack.attack_search import attack_eval
from attack.attack_base import RepeatAttack

# import tensorflow as tf
# tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

# CIFAR eps=4/255 test
device = 'cuda'
epsilon = 4/255
norm = ep.inf
# epsilon = 32/255
# norm = 2
ntest = 10000

torch.manual_seed(0)

# dataset
bounds = (0, 1)

# meta attack scheme

fgsm = LinfFastGradientAttack()
pgd = LinfProjectedGradientDescentAttack({'loss': "logit", "targeted": False})
df = LinfDeepFoolAttack()
cw = LinfCarliniWagnerAttack()

fgsm_baseline = RepeatAttack(fgsm, 100)
pgd_baseline = RepeatAttack(pgd, 8)
df_baseline = df
cw_baseline = cw

attack_list = [fgsm_baseline, pgd_baseline, df_baseline, cw_baseline]

apgd_ce = LinfAPGDAttack()
apgd_dlr = LinfAPGDAttack({"loss": 'dlr'})
fab = LinfFABAttack()
sqr = LinfSquareAttack()
aa = LinfAutoAttack()

model_class_list = [Model1(load=True), Model2(load=True), Model3(load=True), Model4(load=True), Model5(load=True),
                    Model6(load=True), Model7(load=True), Model8(load=True), Model9(load=True), Model10(load=True)]

for f_net in model_class_list:
    attack_eval([pgd_baseline], f_net, epsilon, norm, ntest, dataset='cifar10', use_vertex_removal=False)

# for f_net in model_class_list:
#     attack_eval(attack_list, f_net, epsilon, norm, ntest, dataset='cifar10', use_vertex_removal=True)

