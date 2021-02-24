""" This file defines the search space for attack search.
"""

from attack.attack_pool import *
from attack.attack_base import *
from hyperopt import hp


class search_space:
    def __init__(self, flags):
        self.flags = flags or {}

    def define_space(self, config, defense_property):
        raise NotImplementedError


class Linf_search_space(search_space):
    def __init__(self, flags=None):
        super().__init__(flags)

    def define_space(self, config, defense_property):
        if "use_loss" in self.flags and not self.flags["use_loss"]:
            set_targeted = False
            loss = hp.choice('loss', ['ce', 'l1'])
            loss_nologit = hp.choice('loss_nologit', ['ce', 'l1'])
            use_softmax = False
            use_loss_diff = False
        else:
            set_targeted = hp.choice('set_targeted', [True, False])
            if defense_property.detector:
                loss = hp.choice('loss', ['ce', 'dlr', 'hinge', 'l1', 'logit'])
            else:
                loss = hp.choice('loss', ['ce', 'dlr', 'hinge', 'l1'])
            loss_nologit = hp.choice('loss_nologit', ['ce', 'dlr', 'hinge', 'l1'])
            use_softmax = hp.choice('use_softmax', [True, False])
            use_loss_diff = hp.choice('loss_diff', [True, False])
        n_target_classes = hp.randint('n_target_classes', 1, 9)
        n_target_classes_expensive = hp.randint('n_target_classes_expensive', 1, 3)
        # attack param
        PGD_repeat = hp.randint('PGD_repeat', 1, 10)
        PGD_steps = hp.randint('PGD_steps', 20, 200)
        PGD_rel_stepsize = hp.loguniform('PGD_rel_stepsize', np.log(1 / 1000), np.log(1))
        PGD_EOT = 1
        FGSM_repeat = hp.loguniform('FGSM_repeat', np.log(10), np.log(10000))
        FGSM_EOT = 1
        DF_repeat = 1
        DF_candidates = hp.randint('candidates', 2, 10)
        CW_confidence = hp.uniform('CW_confidence', 0, 0.1)
        CW_max_iter = hp.randint('CW_max_iter', 20, 200)
        CW_learning_rate = hp.loguniform('CW_learning_rate', np.log(0.0001), np.log(0.01))
        CW_repeat = 1
        CW_max_halving = hp.randint('CW_max_halving', 5, 15)
        CW_max_doubling = hp.randint('CW_max_doubling', 5, 15)
        FAB_n_restarts = hp.randint('FAB_n_restarts', 1, 10)
        FAB_n_iter = hp.randint('FAB_n_iter', 10, 200)
        FAB_eta = hp.uniform('FAB_eta', 1, 1.2)
        FAB_beta = hp.uniform('FAB_beta', 0.7, 1)
        SQR_p_init = hp.uniform('SQR_p_init', .5, .9)
        SQR_n_queries = hp.randint('SQR_n_queries', 2000, 8000)
        SQR_n_restarts = hp.randint('SQR_n_restarts', 1, 3)
        NES_steps = hp.randint('NES_steps', 20, 80)
        NES_samples = hp.loguniform('NES_samples', np.log(200), np.log(5000))
        NES_rel_stepsize = hp.loguniform('NES_rel_stepsize', np.log(0.01), np.log(0.1))
        APGD_rho = hp.uniform('APGD_rho', 0.5, 0.9)
        APGD_EOT = 1
        APGD_n_restarts = hp.randint('APGD_n_restarts', 1, 10)
        APGD_n_iter = hp.randint('APGD_n_iter', 20, 200)
        BB_steps = hp.randint('BB_steps', 20, 100)
        BB_momentum = hp.uniform('BB_momentum', 0, 1)

        if defense_property.is_random:
            FGSM_EOT = hp.randint('FGSM_EOT', 1, 200)
            PGD_EOT = hp.randint('PGD_EOT', 1, 40)
            APGD_EOT = hp.randint('APGD_EOT', 1, 40)
            PGD_repeat = 1
            APGD_n_restarts = 1
            FGSM_repeat = 1
            SQR_n_restarts = 1
            FAB_n_restarts = 1



        FGSM_dict = {'attack': LinfFastGradientAttack(),
                     'loss': loss,
                     'set_targeted': set_targeted,
                     'n_target_classes': n_target_classes,
                     'mod': {'softmax': use_softmax, 'loss_diff': use_loss_diff},
                     'repeat': FGSM_repeat,
                     'EOT': FGSM_EOT,
                     }
        PGD_dict = {'attack': LinfProjectedGradientDescentAttack(),
                    'loss': loss,
                    'set_targeted': set_targeted,
                    'n_target_classes': n_target_classes,
                    'mod': {'softmax': use_softmax, 'loss_diff': use_loss_diff},
                    'repeat': PGD_repeat,
                    'EOT': PGD_EOT,
                    'steps': PGD_steps,
                    'rel_stepsize': PGD_rel_stepsize,
                    }
        DF_dict = {'attack': LinfDeepFoolAttack(),
                   'loss': loss_nologit,
                   'set_targeted': set_targeted,
                   'candidates': DF_candidates,
                   'mod': {'softmax': use_softmax, 'loss_diff': use_loss_diff},
                   'repeat': DF_repeat,
                   }
        CW_dict = {'attack': LinfCarliniWagnerAttack(),
                    'set_targeted': set_targeted,
                    'n_target_classes': n_target_classes,
                    'repeat': CW_repeat,
                   'confidence': CW_confidence,
                   'max_iter': CW_max_iter,
                   'learning_rate': CW_learning_rate,
                   'max_halving': CW_max_halving,
                   'max_doubling': CW_max_doubling,}
        FAB_dict = {'attack': LinfFABAttack(),
                    'set_targeted': set_targeted,
                    'n_target_classes': n_target_classes,
                    "n_restarts": FAB_n_restarts,  # repeat
                    "n_iter": FAB_n_iter,
                    "beta": FAB_beta,
                    "eta": FAB_eta,
                    }
        SQR_dict = {'attack': LinfSquareAttack(),
                    "loss": loss,
                    'set_targeted': set_targeted,
                    'n_target_classes': n_target_classes_expensive,
                    'mod': {'softmax': use_softmax, 'loss_diff': use_loss_diff},
                    "p_init": SQR_p_init,
                    "n_queries": SQR_n_queries,
                    "n_restarts": SQR_n_restarts,  # repeat
                    }
        NES_dict = {"attack": LinfNaturalEvolutionStrategyAttack(),
                    "loss": loss,
                    'set_targeted': set_targeted,
                    'n_target_classes': n_target_classes_expensive,
                    'mod': {'softmax': use_softmax, 'loss_diff': use_loss_diff},
                    'steps': NES_steps,
                    'rel_stepsize': NES_rel_stepsize,
                    'n_samples': NES_samples,
                    'EOT': 1,
                    'repeat': 1,
                    'parallel': False,
                    }
        APGD_dict = {"attack": LinfAPGDAttack(),
                     "loss": loss,
                     'set_targeted': set_targeted,
                     'n_target_classes': n_target_classes,
                     'mod': {'softmax': use_softmax, 'loss_diff': use_loss_diff},
                     "eot_iter": APGD_EOT,
                     "n_restarts": APGD_n_restarts,  # repeat
                     "n_iter": APGD_n_iter,
                     "rho": APGD_rho,
                     }
        BB_dict = {'attack': LinfBrendelBethgeAttack(),
                   "overshoot": 1.1,
                   "steps": BB_steps,
                   "lr": 0.01,
                   "lr_decay": 0.5,
                   "lr_num_decay": 20,
                   "momentum": BB_momentum, }

        attack_space = [FGSM_dict, PGD_dict, DF_dict, CW_dict, FAB_dict, SQR_dict, NES_dict]

        flags = self.flags
        if "APGD" in flags and flags["APGD"]:
            attack_space.append(APGD_dict)

        space = {'config': config, 'param': hp.choice('attack_type', attack_space)}
        return space
