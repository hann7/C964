import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from app.server.model.utils.preprocess_data import process_and_split_data

labels_file_path = '../../../data/sample_labels.csv'
images_path = '../../../data/sample_dataset_processed'

X_train, X_test, y_train, y_test = process_and_split_data(labels_file_path, images_path)


base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# creates a custom layer for our own use with the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# saves the best performing model even if training is not complete
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    './saved_models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min'
)

# stops training if the model has not improved for 5 epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor='val_loss'
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=16,
    callbacks=[checkpoint, early_stop]
)

# saves final model after all training
model.save('./saved_models/diabetic_retinopathy_model.h5')