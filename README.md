# code for fredis

Code for 'FREDIS: A Fusion Framework of Refinemnet and Disambiguation for Unreliable Partial Label Learning' (ICML'23)

# Running Environment
Requirements: 
Python 3.6.9, 
numpy 1.19.5, 
torch 1.9.1,
torchvision 0.10.1.

or 

```
conda env create -f environment.yaml
```

# Run the Demo
You need to:
1. Download the datasets into './data/'.
2. Run the following example:
```
python main.py --dataset cifar10 --noisy_rate 0.3 --partial_rate 0.3

```


