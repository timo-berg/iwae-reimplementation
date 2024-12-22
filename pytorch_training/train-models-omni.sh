#!/bin/bash

# Run VAE with num_samples=1 and num_stochastic_layers=2
python main_train_omni.py --model VAE --num_stochastic_layer 2 --num_samples 1

# Run IWAE with num_samples=5 and num_stochastic_layers=2
python main_train_omni.py --model IWAE --num_stochastic_layer 2 --num_samples 5

# Run IWAE with num_samples=50 and num_stochastic_layers=2
python main_train_omni.py --model IWAE --num_stochastic_layer 2 --num_samples 50

# Run VAE with num_samples=5 and num_stochastic_layers=2
python main_train_omni.py --model VAE --num_stochastic_layer 2 --num_samples 5

# Run VAE with num_samples=50 and num_stochastic_layers=2
python main_train_omni.py --model VAE --num_stochastic_layer 2 --num_samples 50