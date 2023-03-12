
# Test

Run main.py

Function "train_optimizer" does the training, the trained network is saved at "path".

Function "test_optimizer" does testing.

Function "forward_pass" does the network forward pass and is equivalent to calling "do_fit" with parameter should_train=False. do_fit has some extra operations (gradient computation, etc.) that are only required during training.

Class Optimizer is the LSTM neural network.

Class OptimizeeMISO represents the decision variable to be optimized (i.e., the beamformer matrix).

SumRateObjective is the optimization objective.

Set USE_CUDA=True to use gpu, otherwise use CPU.

Inspired by: https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent/blob/master/Grad_buffer.ipynb
