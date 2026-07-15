import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. Prepare your data
data_path = os.path.abspath('data') 

# --- THE UPDATE: Artificial Data Multiplier ---
# We added rotation, shifting, zooming, and flipping to help the AI learn 
# from your 20 images as if there were thousands.
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Keeps 20% of images aside to test the AI during training
)

print("Loading training data...")
train_data = datagen.flow_from_directory(
    data_path, 
    target_size=(224, 224), 
    batch_size=8, 
    classes=['safe', 'prohibited'], # Forces Safe=0, Prohibited=1
    subset='training'
)

print("Loading validation data...")
val_data = datagen.flow_from_directory(
    data_path, 
    target_size=(224, 224), 
    batch_size=8, 
    classes=['safe', 'prohibited'], 
    subset='validation'
)

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # Added Dropout

# 2. Use a "Pre-trained Brain" (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- THE CRITICAL FIX ---
# Freeze the base model so your small dataset doesn't destroy its memory!
base_model.trainable = False 
# ------------------------

# 3. Add your custom "tail" to the brain
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# --- ANTI-MEMORIZATION FIX ---
# This randomly turns off 50% of the neurons during training so it CANNOT memorize images.
x = Dropout(0.5)(x) 
# -----------------------------

predictions = Dense(2, activation='softmax')(x) # 2 classes: Safe/Prohibited

model = Model(inputs=base_model.input, outputs=predictions) 

# 4. Train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Notice we added validation_data here so you can see how well it's learning!
model.fit(train_data, validation_data=val_data, epochs=10) 
# Bumped epochs to 10 so it has time to learn the new augmented images

# 5. Save the brain
model.save('content_model.h5')
print("Model trained and saved as content_model.h5.") 