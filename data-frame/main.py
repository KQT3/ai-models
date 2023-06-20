from builtins import print

import pandas as pd
import tensorflow as tf
import pprint


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(self)

        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.seq = tf.keras.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def adapt(self, inputs):
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)

    def call(self, inputs):
        inputs = stack_dict(inputs)
        result = self.seq(inputs)

        return result


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


def main():
    SHUFFLE_BUFFER = 500
    BATCH_SIZE = 2

    csv_file = tf.keras.utils.get_file('heart.csv',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
    df = pd.read_csv(csv_file)

    print(df.head())
    print(df.dtypes)

    target = df.pop('target')

    numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
    numeric_features = df[numeric_feature_names]
    print(numeric_features.head())
    print(tf.convert_to_tensor(numeric_features))

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(numeric_features)

    print(normalizer(numeric_features.iloc[:3]))

    def get_basic_model():
        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    model = get_basic_model()
    model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

    numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

    for row in numeric_dataset.take(3):
        print(row)

    numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

    model = get_basic_model()
    model.fit(numeric_batches, epochs=15)

    numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))

    for row in numeric_dict_ds.take(3):
        print(row)

    model = MyModel()

    model.adapt(dict(numeric_features))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  run_eagerly=True)

    model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

    numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    model.fit(numeric_dict_batches, epochs=5)

    model.predict(dict(numeric_features.iloc[:3]))

    inputs = {}
    for name, column in numeric_features.items():
        inputs[name] = tf.keras.Input(
            shape=(1,), name=name, dtype=tf.float32)

    print(inputs)

    x = stack_dict(inputs, fun=tf.concat)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features)))

    x = normalizer(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, x)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  run_eagerly=True)

    tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)

    model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

    numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    model.fit(numeric_dict_batches, epochs=5)

    binary_feature_names = ['sex', 'fbs', 'exang']

    categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

    inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif (name in categorical_feature_names or
              name in binary_feature_names):
            dtype = tf.int64
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

    print(inputs)

    preprocessed = []

    for name in binary_feature_names:
        inp = inputs[name]
        inp = inp[:, tf.newaxis]
        float_value = tf.cast(inp, tf.float32)
        preprocessed.append(float_value)

    print(preprocessed)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features)))

    numeric_inputs = {}
    for name in numeric_feature_names:
        numeric_inputs[name] = inputs[name]

    numeric_inputs = stack_dict(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    preprocessed.append(numeric_normalized)

    print(preprocessed)

    vocab = ['a', 'b', 'c']
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
    lookup(['c', 'a', 'a', 'b', 'zzz'])

    vocab = [1, 4, 7, 99]
    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

    lookup([-1, 4, 1])

    for name in categorical_feature_names:
        vocab = sorted(set(df[name]))
        print(f'name: {name}')
        print(f'vocab: {vocab}\n')

        if type(vocab[0]) is str:
            lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
        else:
            lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

        x = inputs[name][:, tf.newaxis]
        x = lookup(x)
        preprocessed.append(x)
    print(preprocessed)

    preprocessed_result = tf.concat(preprocessed, axis=-1)
    print(preprocessed_result)

    preprocessor = tf.keras.Model(inputs, preprocessed_result)
    tf.keras.utils.plot_model(preprocessor, rankdir="LR", show_shapes=True)
    print(preprocessor(dict(df.iloc[:1])))

    body = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    print(inputs)

    x = preprocessor(inputs)
    print(x)

    result = body(x)
    print(result)

    model = tf.keras.Model(inputs, result)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)
    print(history)

    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    ds = ds.batch(BATCH_SIZE)

    for x, y in ds.take(1):
        pprint.pprint(x)
        print()
        print(y)

    history = model.fit(ds, epochs=5)
    print(history)


if __name__ == '__main__':
    main()
