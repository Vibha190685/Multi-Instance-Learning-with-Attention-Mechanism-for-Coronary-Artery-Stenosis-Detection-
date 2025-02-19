import os
os.environ['TF_USE_LEGACY_KERAS']='1'
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import pprint
from tensorflow.python import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201,VGG16,EfficientNetB0,EfficientNetB3
from tensorflow.keras.layers import TimeDistributed, Reshape, Activation, Dot, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, TimeDistributed,LSTM, GRU,LayerNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.utils import to_categorical


class PositionalEncodingLayer(Layer):
    def __init__(self, max_position, feature_dim, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.max_position = max_position  # Store max_position for get_config
        self.feature_dim = feature_dim  # Store feature_dim for get_config
        self.positional_encoding = self.add_weight(
            shape=(max_position, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='positional_encoding'
        )

    def call(self, inputs):
        # Assuming input shape is (batch_size, num_slices, feature_dim)
        positions = tf.range(tf.shape(inputs)[1])  # Get the positions from the input shape
        # Add positional encodings to the inputs
        return inputs + tf.gather(self.positional_encoding, positions)

    def get_config(self):
        # Return a dictionary of the configuration for the layer
        config = super(PositionalEncodingLayer, self).get_config()
        config.update({
            'max_position': self.max_position,
            'feature_dim': self.feature_dim,
        })
        return config


    
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1] if input_shape[0][-1] is not None else 128  # Handle None case
        print(f"Feature dimension: {feature_dim}")
        self.attention_weights = self.add_weight(name='attention_weights',
                                                 shape=(feature_dim, 1),
                                                 initializer='glorot_uniform',
                                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        multi_head_attention, _ = inputs
        # Calculate attention logits
        attention_logits = tf.matmul(multi_head_attention, self.attention_weights)
        # Calculate attention weights (probabilities)
        attention_weights = tf.nn.softmax(attention_logits, axis=1)
        # Calculate the weighted sum of features using the attention weights
        weighted_sum = tf.reduce_sum(multi_head_attention * attention_weights, axis=1)
        # Compute final output by averaging the weighted sum
        final_output = tf.reduce_mean(weighted_sum, axis=1, keepdims=True)
        final_output = tf.expand_dims(final_output, axis=1)

        # Always return both final output and attention weights
        return final_output, attention_weights

    # Add the `get_config()` method for serialization support
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


class CustomMILDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, image_size, batch_size, is_training=True, num_workers=4):
        self.directory = directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.is_training = is_training
        self.patients = self._load_patients()
        #print(f'Initialized patients: {self.patients}')  # Debugging line
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            #rotation_range=15,
            #width_shift_range=0.1,
            #height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            #horizontal_flip=True,
            fill_mode='nearest' 
            ) if is_training else ImageDataGenerator()
        self.on_epoch_end()
        self.num_positive_bags, self.num_negative_bags = self._count_bags()
 
    def _count_bags(self):
        num_positive = 0
        num_negative = 0
        for patient_images in self.patients:
            bag_label = self._get_bag_label(patient_images)
            if bag_label == 1:
                num_positive += 1
            else:
                num_negative += 1
        return num_positive, num_negative
    def _load_patients(self):
        class_folders = os.listdir(self.directory)
        #print(f'Class folders found: {class_folders}')  # Debugging line
        patients = []
        for class_folder in class_folders:
            class_path = os.path.join(self.directory, class_folder)
            if os.path.isdir(class_path):
                patient_folders = os.listdir(class_path)
                #print(f'Patient folders in {class_folder}: {patient_folders}')  # Debugging line
                for patient_folder in patient_folders:
                    patient_path = os.path.join(class_path, patient_folder)
                    if os.path.isdir(patient_path):
                        image_paths = [os.path.join(patient_path, img) for img in os.listdir(patient_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if image_paths:
                            patients.append(image_paths)
                            #print(f'Loaded patient: {patient_folder} with images: {image_paths}')
                        else:
                            print(f'No images found in {patient_path}')
                    else:
                        print(f'{patient_path} is not a directory')
            else:
                print(f'{class_path} is not a directory')
        #print(f'Loaded patient: {patients}')    
        return patients
    def __len__(self):
        #return int(np.floor(len(self.patients) //self.batch_size))
        return len(self.patients) // self.batch_size
        #return int(np.ceil(len(self.patients) / self.batch_size))
    def __getitem__(self, index):
        # Extract patients for the current batch
        batch_patients = self.patients[index * self.batch_size:(index + 1) * self.batch_size]
        images, bag_labels = [], []
        for patient_images in batch_patients:
            # Load images and bag labels for each patient
            patient_images_data = [self._load_image(img_path) for img_path in patient_images]
            images.append(patient_images_data)
            bag_label = self._get_bag_label(patient_images)
            bag_labels.append(bag_label)
        num_bags = len(batch_patients)
        # Convert lists to numpy arrays
        images_array = np.array(images)
        bag_labels_array = np.array(bag_labels) 
        bag_labels_array = bag_labels_array.reshape(-1, 1)
 
        #return (images_array, bag_labels_array), bag_labels_array
        return (images_array, bag_labels_array), bag_labels_array
 

    def _load_image(self, path):
        if not os.path.isfile(path):
            print(f'Image file not found: {path}')
            return np.zeros((self.image_size[0], self.image_size[1], 3))  # Return a placeholder if file is missing
        image = tf.keras.preprocessing.image.load_img(path, target_size=self.image_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        #image = image / 255.0
 
        # Apply data augmentation only if is_training is True
        if self.is_training:
            image = self.datagen.random_transform(image)  # Augment the image
        #print(f'Loaded image: {path}')
        return image
        
    def _get_bag_label(self, image_paths):
        labels = [self._get_image_label(img_path) for img_path in image_paths]
        #print(f'Labels for images in bag: {labels}')
        #return int(any(label == 1 for label in labels))  # Adjust based on your actual label criteria
        label = int(any(label == 1 for label in labels))
        # Convert the label to one-hot encoding
        num_classes = 2  # Assuming binary classification
        one_hot_label = np.zeros(num_classes)
        one_hot_label[label] = 1
        return  label
    def _get_image_label(self, image_path):
        return 1 if 'class_2' in os.path.basename(os.path.dirname(os.path.dirname(image_path))) else 0
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.patients)


    

def MIL_Pos_MHA_Att(input_shape=(None, 256, 256, 3), base_model_name='VGG16', num_attention_heads=4, 
                          learning_rate=0.0001, dropout_rate=0.2, freeze_layers=15, lambda_value=0.01):
    # Choose the base model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    elif base_model_name == 'DenseNet169':
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape[1:]) 
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    elif base_model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    else:
        raise ValueError(f"Base model {base_model_name} not supported yet.")
 
    # Freeze specified number of layers
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
 
    # Define inputs
    image_input = Input(shape=input_shape)  # Input for the bag of images
    bag_label = Input(shape=(1,))  # Bag label (0 or 1)
 
    # Instance feature extraction
    instance_features = TimeDistributed(base_model)(image_input)
    pooled_features = TimeDistributed(GlobalAveragePooling2D())(instance_features)
    layer_norm_layer = TimeDistributed(LayerNormalization())(pooled_features)
    dropout_layer = TimeDistributed(Dropout(dropout_rate))(layer_norm_layer)
 
    # Dynamically calculate key_dim from pooled_features
    key_dim = dropout_layer.shape[-1]  # Extract feature dimension from pooled features
    print(f"Calculated key_dim: {key_dim}")
 
    # Apply Multi-Head Attention
    positional_encoding = PositionalEncodingLayer(max_position=36, feature_dim=dropout_layer.shape[-1])(dropout_layer)
    multi_head_attention = MultiHeadAttention(num_heads=num_attention_heads, key_dim=key_dim)(positional_encoding, positional_encoding)
 
    # Use the custom attention layer
    #weighted_sum = AttentionLayer([multi_head_attention, bag_label])
    #attention_layer = AttentionLayer(return_attention=True)  # Set to True to return attention weights
    #weighted_sum, attention_weights = attention_layer([multi_head_attention, bag_label])
    #weighted_sum,attention_weights = AttentionLayer()([multi_head_attention, bag_label])
    attention_layer = AttentionLayer()  # No need to pass return_attention anymore

    # When using the layer
    weighted_sum, attention_weights = attention_layer([multi_head_attention, bag_label])
    # Bag-level prediction
    bag_prediction = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_value))(weighted_sum)
 
    # Define and compile model
    model = Model(inputs=[image_input, bag_label], outputs=bag_prediction)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


