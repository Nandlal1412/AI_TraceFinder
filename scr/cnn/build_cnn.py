import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_dual_branch(input_shape_img=(256,256,3), input_shape_noise=(256,256,1), num_classes=11):
    # Branch 1: Official
    img_input = Input(shape=input_shape_img, name="official_input")
    x1 = layers.Conv2D(32, (3,3), activation="relu")(img_input)
    x1 = layers.MaxPooling2D((2,2))(x1)
    x1 = layers.Conv2D(64, (3,3), activation="relu")(x1)
    x1 = layers.MaxPooling2D((2,2))(x1)
    x1 = layers.Conv2D(128, (3,3), activation="relu")(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Branch 2: Noise
    noise_input = Input(shape=input_shape_noise, name="noise_input")
    x2 = layers.Conv2D(32, (3,3), activation="relu")(noise_input)
    x2 = layers.MaxPooling2D((2,2))(x2)
    x2 = layers.Conv2D(64, (3,3), activation="relu")(x2)
    x2 = layers.MaxPooling2D((2,2))(x2)
    x2 = layers.Conv2D(128, (3,3), activation="relu")(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Fusion
    combined = layers.Concatenate()([x1, x2])
    z = layers.Dense(256, activation="relu")(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation="softmax")(z)

    model = models.Model(inputs=[img_input, noise_input], outputs=output)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    return model
