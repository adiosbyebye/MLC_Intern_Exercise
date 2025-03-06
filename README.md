# MLC_Intern_Exercise

JPM 2025 Time Series & Reinforcement Learning internship exercise code repository for Yan Kin (Chris) Chi


Question 1:
We subdivide the sampling task into three parts for easy debugging:

Part 1
- run ./Question1/part1/generate_samples.py to generate training samples for the L-HNN (parameters stored in config_training_data.yaml, remember to choose a distribution)
- run ./Question1/part1/hnn_train.py to train the L-HNN (parameters stored in config.yaml)
- run ./Question1/part1/hnn_nuts.py to generate samples from the Posterior

Part 2
- run ./Question1/part2/generate_samples.py to generate training samples for the L-HNN (parameters stored in config_training_data.yaml, distribution fixed to adhere to Gaussian with fixed/random effects)
- run ./Question1/part2/train_hnn.py to train the L-HNN (parameters stored in config.yaml)
- run ./Question1/part2/hnn_nuts.py to generate samples from the Posterior
