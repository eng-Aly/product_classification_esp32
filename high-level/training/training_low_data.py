

import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Directory to save images
SCRIPT_DIR = Path(__file__).resolve().parent


IMG_SIZE = 96
ALPHA = 0.35
NUM_CLASSES = 2          
BATCH_SIZE = 16        
DATA_DIR = SAVE_DIR = SCRIPT_DIR.parent / "data" 
VAL_SPLIT = 0.2          


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

augment = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)


preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=ALPHA,
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
backbone.trainable = False  

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = backbone(inputs, training=False)
x = layers.Dropout(0.3)(x)          
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop],
)

model.save("product_classifier.keras")
print("Saved -> product_classifier.keras")
print(f"Final val accuracy: {max(history.history['val_accuracy']):.3f}")
