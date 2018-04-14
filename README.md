### Lie Detector on CSC Deceptive Speech Dataset using simple RNN with TensorFlow

Implemented a simple RNN lie detector on the CSC Deceptive Speech Dataset using [TensorFlow 1.7 Estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator).

I tried modeling the data in two different ways; first (per frame method), pass in a MFCC vector with each row as a time step to the RNN, for instance, if the MFCC vector has dimension (28000,13), then the input for the RNN model is (28000, batch_size, 13); second , pass in the MFCC vector such that the whole vector is treated as the same time step, i.e. (1, 1, 28000 * 13). The two method can be selected using flag `--per_frame`.

The the MFCC vectors have different length, in order to pass in the vectors into TensorFlow's Estimator API, I had to add paddings to the vectors that are shorter than the longest vector, and use the `sequence_length` parameter in `dynamic_rnn to` ignore the padding. The model support the basic RNN, LSTM and GRU which can be specified with flag `--model`. When using the `per frame` method, the last output state from `dynamic_rnn` is then taken to apply dropout with flag `--dropout`, and pass to the output layer.

#### Dataset
- [CSC Deceptive Speech Dataset](https://catalog.ldc.upenn.edu/LDC2013S09)
- Also available at `data` in the repository

#### How to run
`python3 lie-detector.py --lr=0.001 ---epochs=5 --model=LSTM --data_dir=data --per_frame=True`
