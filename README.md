# MLC_Intern_Exercise

JPM 2025 Time Series & Reinforcement Learning internship exercise code repository for Yan Kin (Chris) Chi


Question 1:
We subdivide the sampling task into three parts for easy debugging:
- run ./Question1/partgenerate_samples.py to generate training samples for the L-HNN (parameters stored in config_training_data.yaml)
- run ./Question1/hnn_train.py to train the L-HNN (parameters stored in config.yaml)
- run ./Question1/hnn_nuts.py to generate samples from the Posterior
