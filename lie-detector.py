import os
import fnmatch
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell

tf.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.flags.DEFINE_integer('epochs', 5, 'number of epoch')
tf.flags.DEFINE_integer("hidden_size", 256, "hidden size for each layer")
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.flags.DEFINE_integer('eval_every', 200,
                        'evaluation after number of train steps')
tf.flags.DEFINE_bool('normalize', False, 'normalize feature data')
tf.flags.DEFINE_float('dropout', 0.2, 'dropout rate')
tf.flags.DEFINE_string('model', 'GRU', 'RNN, GRU or LSTM')
tf.flags.DEFINE_string('data_dir', 'data', 'directory of original data files')
tf.flags.DEFINE_string('log_dir', 'tmp/runs/', 'directory to save log file')
tf.flags.DEFINE_bool('per_frame', True, 'RNN on per frame (row) data instead '
                                        'of taking the whole MFCC vector ')


FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


class Params(object):
    """ hyper-parameters """
    lr = FLAGS.lr
    epochs = FLAGS.epochs
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    train_steps = 0
    eval_steps = 0
    eval_every = FLAGS.eval_every
    normalize = FLAGS.normalize
    dropout = FLAGS.dropout
    model = FLAGS.model
    data_dir = FLAGS.data_dir
    log_dir = FLAGS.log_dir
    num_classes = 3
    feature_length = 13
    max_length = 0
    per_frame = FLAGS.per_frame


def generate_data(params):
    """ Extract data and transcript from FLAGS.data_dir
    Note: 0 indicate True, 1 indicate Lie Up, 2 indicate Lie Down for labels
    """
    if not os.path.exists(params.data_dir):
        print("Data directory %s not found" % params.data_dir)
        exit()
    features = []
    labels = []
    sequence_length = []
    for subdir, dirs, files in os.walk(params.data_dir):
        for speaker in dirs:
            with open(os.path.join(
                    params.data_dir, speaker, 'transcripts.txt'), 'r') as f:
                transcripts = f.readlines()
            if not transcripts:
                continue
            files = sorted(fnmatch.filter(os.listdir(
                os.path.join(params.data_dir, speaker)), '*npy'))

            assert len(transcripts) == len(files)

            for i in range(len(transcripts)):
                # read MFCC vector from npy file
                features.append(np.load(
                    os.path.join(FLAGS.data_dir, speaker, files[i])))
                # read label from transcripts
                label = transcripts[i].split()[1]
                if label.startswith('T'):
                    labels.append(0)
                elif label.startswith('LU'):
                    labels.append(1)
                elif label.startswith('LD'):
                    labels.append(2)
                else:
                    print("Incorrect label: %s" % label)
                    exit()

    # add padding to create equal length MFCC vectors
    params.max_length = max([feature.shape[0] for feature in features])

    for i in range(len(features)):
        # pad vectors
        padding = params.max_length - features[i].shape[0]
        sequence_length.append(features[i].shape[0])
        features[i] = np.vstack(
            (features[i], np.zeros(shape=(padding, params.feature_length))))


    # convert to ndarray
    features, labels = np.asarray(features), np.asarray(labels)

    # normalize features
    if params.normalize:
        shape = features.shape
        # normalize function only takes 2D matrix
        features = np.reshape(features, newshape=(shape[0], shape[1] * shape[2]))
        features = normalize(features, norm='l2')
        features = np.reshape(features, newshape=shape)

    assert features.shape[0] == labels.shape[0] == len(sequence_length)

    # randomly shuffle data
    features, labels, sequence_length = \
        shuffle(features, labels, sequence_length, random_state=1)

    return features, labels, sequence_length


def metric_fn(labels, predictions):
    """ Metric function for evaluations"""
    return {'eval_accuracy': tf.metrics.accuracy(labels, predictions),
            'eval_precision': tf.metrics.precision(labels, predictions),
            'eval_recall': tf.metrics.recall(labels, predictions)}


def rnn(features, mode, params):
    """ Recurrent model """
    if params.model == "LSTM":
        cell = BasicLSTMCell(params.hidden_size)
    elif params.model == "GRU":
        cell = GRUCell(params.hidden_size)
    else:
        cell = BasicRNNCell(params.hidden_size)

    initial_state = cell.zero_state(params.batch_size, dtype=tf.float64)

    if params.per_frame:
        # convert input from (batch_size, max_time, ...) to
        # (max_time, batch_size, ...)
        inputs = tf.transpose(features['feature'], [1, 0, 2])

        sequence_length = tf.reshape(
            features['sequence_length'],
            shape=(params.batch_size,)
        )

        outputs, state = tf.nn.dynamic_rnn(
            cell,
            inputs=inputs,
            initial_state=initial_state,
            sequence_length=sequence_length,
            time_major=True
        )

        # get output from the last state
        outputs = outputs[features['sequence_length'][0] - 1]
    else:
        # reshape MFCC vector to fit in one time step
        inputs = tf.reshape(
            features['feature'],
            shape=(1, params.batch_size, params.max_length * params.feature_length)
        )

        outputs, state = tf.nn.dynamic_rnn(
            cell,
            inputs=inputs,
            initial_state=initial_state,
            time_major=True
        )

        outputs = tf.reshape(
            outputs,
            shape=(params.batch_size, params.hidden_size)
        )

    # apply dropout
    dropout = tf.layers.dropout(
        outputs,
        rate=params.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(
        dropout,
        units=params.num_classes,
        activation=None
    )

    return logits


def model_fn(features, labels, mode, params):
    """ Estimator model function"""
    logits = rnn(features, mode, params)

    predictions = tf.argmax(tf.nn.softmax(logits), axis=-1)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
    )

    train_op = tf.train.AdamOptimizer(params.lr).minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    # metrics summary
    tf.summary.text('prediction', tf.as_string(predictions))
    tf.summary.text('label', tf.as_string(labels))
    accuracy = tf.metrics.accuracy(labels, predictions)
    tf.summary.scalar('training_accuracy', accuracy[1])
    precision = tf.metrics.precision(labels, predictions)
    tf.summary.scalar('training_precision', precision[1])
    recall = tf.metrics.recall(labels, predictions)
    tf.summary.scalar('training_recall', recall[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=metric_fn(labels, predictions)
        )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def main():
    # initialize model parameters
    params = Params()

    # check if log directory exist
    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)

    features, labels, sequence_length = generate_data(params)
    # index of training and testing data split
    split = int(len(labels) * 0.8)

    # calculate the amount of train and test steps
    params.train_steps = int(split / params.batch_size) * params.epochs
    params.eval_steps = int((len(features) - split) / params.batch_size)

    def train_input_fn(params):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'feature': features[:split],
                'sequence_length': sequence_length[:split]
            },
            labels[:split]
        ))
        dataset = dataset.repeat().batch(params.batch_size)
        x, y = dataset.make_one_shot_iterator().get_next()
        return x, y

    def eval_input_fn(params):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'feature': features[split:],
                'sequence_length': sequence_length[split:]
            },
            labels[split:]
        ))
        dataset = dataset.batch(params.batch_size)
        x, y = dataset.make_one_shot_iterator().get_next()
        return x, y

    # setup Estimator configuration
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=params.eval_every
    )

    # define Estimator class for model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params.log_dir,
        config=config,
        params=params
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.train_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=params.eval_steps
    )

    # train and evaluate model
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )


if __name__ == "__main__":
    main()
