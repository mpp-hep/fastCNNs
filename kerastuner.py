# Commented out IPython magic to ensure Python compatibility.
# !pip install -U keras-tuner
# !export KERASTUNER_TUNER_ID="tuner0"
# !export KERASTUNER_ORACLE_IP="127.0.0.1"
# !export KERASTUNER_ORACLE_PORT="8000"
import os
import kerastuner as kt

import tensorflow as tf
print("Tensorflow version " + tf.__version__)
import tensorflow_datasets as tfds
AUTO = tf.data.experimental.AUTOTUNE

strategy = tf.distribute.MirroredStrategy() 

BATCH_SIZE = 16*strategy.num_replicas_in_sync  # A TPU has 8 cores so this will be 128

def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    image = item['image']
    label = item['label']
    label = tf.one_hot(tf.squeeze(label), 10)
    image = tf.dtypes.cast(image, 'float32') / 255.
    return image, label
    
ds_data = tfds.load('svhn_cropped')#'gs://tfds-data/datasets')
ds_train, ds_test = ds_data['train'], ds_data['test']
ds_train = ds_train.map(convert_dataset)
ds_train = ds_train.cache()
ds_train = ds_train.repeat()
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTO)
ds_test = ds_test.map(convert_dataset)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTO)



for img_feature, label in ds_train:
      break
print(" --------------INPUT INFO --------------")
print('INPUT img_feature.shape (batch_size, image_height, image_width) =', img_feature.shape)
print('INPUT label.shape (batch_size, number_of_labels) =', label.shape)
IMAGE_SIZE = [img_feature.shape[1], img_feature.shape[2]]

EPOCHS = 20
steps_per_epoch=73257//BATCH_SIZE


"""Then define the model"""
def build_model(hp):
    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3])
    x = inputs
    for i in range(hp.Int('conv_layers', 1, 3, default=3)):
        x = tf.keras.layers.Conv2D( #tf.keras.layers.SeparableConv2D(
            filters=hp.Int('filters_' + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 5),
            activation='relu',
            padding='same')(x)

        if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
            x = tf.keras.layers.MaxPooling2D()(x)
        else:
            x = tf.keras.layers.AveragePooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice('global_flatten', ['max', 'avg']) == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # else:
    #   x = tf.keras.layers.Flatten()
    for i in range(hp.Int('dense_layers', 0, 2, default=1)):
        x = tf.keras.layers.Dense(
            units=hp.Int('neurons_' + str(i), 64, 256, step=64, default=128),
            activation='relu')(x)

    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    model.compile(tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Testing model:")
    model.summary()
    return model

"""Define the keras-tuner object to run the hyperparameter search"""
tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_accuracy',
        max_epochs=100,
        factor=3,
        hyperband_iterations=3,
        distribution_strategy=strategy,
        directory='keras-tune-svhn',
        project_name='v1_conv2D_globPool',
        overwrite=True) #set to false to continue previous scan!


"""Run scan"""
EPOCHS=100
 
 print(" Steps per epoch is {}".format(steps_per_epoch))
 tuner.search(ds_train,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=ds_test,
                 epochs=EPOCHS,
                 callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy')])

"""Get the optimal hyperparameters"""

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of conv layers is {best_hps.get('conv_layers')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

"""Then retrain using the best model!"""

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(ds_train, epochs = 10, validation_data = ds_test)