# Adaptive AutoAttack

"Automatic Discovery of Adaptive Attacks on Adversarial Defenses"

Chengyuan Yao, Pavol Bielik, Petar Tsankov, Martin Vechev

arxiv: https://arxiv.org/abs/2102.11860v1


## Setup:
python version>=3.6
```
1. pip install -r requirement.txt
2. download folder for models from: https://drive.google.com/file/d/1ajCXluAPUPiyLe2ka9i41jj1YQ6u7YX-/view?usp=sharing (2.2GB)
3. unzip the folder and replace the folder zoo/saved_instances
```
## Reproduce main result:
Models need to be selected in the script, and epsilon need to be adjusted. 
```
python bmA_eval.py (A^3 on group A models)
python bmB_eval.py (A^3 on group B models)
python autoattack_eval.py (AutoAttack on defenses)
python aa_detector_eval.py (AutoAttack on defenses with detector)
```

## Reproduce tpe random comparison in supplementary material:
Run TPE vs Random comparison:  
```
python progression_comparison.py --algo 1 --dir=<result_dir_name, e.g tpe_random1>  
```
Run Loss formulation comparison:  
```
python progression_comparison.py --algo 0 --dir=<result_dir_name, eg loss_noloss1>
``` 
Analyze result (need to update the result directory names in the script)
```
python progression_analysis.py --algo <bool> --dual <bool, to plot TPE vs Random and loss formulation together>
```
## Reproduce the scatter plot on attack-scores:
```
python density_plot.py --dir=test
```
