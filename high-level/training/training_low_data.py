

import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Directory to save images
SCRIPT_DIR = Path(__file__).resolve().parent


IMG_SIZE = 96
ALPHA = 0.35
NUM_CLASSES = 2          # update if you add classes
BATCH_SIZE = 16          # small on purpose — small dataset, small batches
DATA_DIR = SAVE_DIR = SCRIPT_DIR.parent / "data" 
VAL_SPLIT = 0.2          # ~12 images/class held out for validation

# ---- 1. Load data ----
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

# ---- 2. Heavy augmentation — critical with only ~60 images/class ----
# Without this, the head will overfit within a few epochs.
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

# MobileNetV2 preprocessing (scales to [-1, 1])
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ---- 3. Build model: frozen backbone + small head ----
# Small head on purpose — a big dense stack will overfit 120 images instantly.
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=ALPHA,
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
backbone.trainable = False  # phase 1: frozen, per earlier discussion

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = backbone(inputs, training=False)
x = layers.Dropout(0.3)(x)          # dropout matters more than usual given dataset size
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---- 4. Train ----
# Early stopping — with this little data, the val loss minimum comes fast.
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
