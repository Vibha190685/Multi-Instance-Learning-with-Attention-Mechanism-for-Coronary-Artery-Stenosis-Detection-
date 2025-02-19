import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
import utilis_file as pc  # Assuming your utility functions are in this module
from tensorflow.keras import backend as K
 
# Base directory and configuration
base_dir = ""   # Base directory for your dataset
image_size = (256, 256)  # Input image size
batch_size = 16
class_names = ['class_1', 'class_2']
#,99,100,499,999,1000,9999
random_states = [42,100,499,999,9999]   # Example of List of random seeds
 
# Directories to save models
save_model_dir = ""

 
# Define dropout rates, freeze layers, and lambda values to test
dropout_rates = [0.3]  # Example dropout rates
freeze_layers = [10]    # Example freeze layer configurations
lambda_values = [0.000001]  # Example lambda values
 
save_model_name = '' ### Name of the model to save
 
# Initialize the multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
 
# Iterate over random seeds
for random_state in random_states:
    # Set directories for each random state
    train_dir = os.path.join(base_dir + str(random_state), 'train')
    val_dir = os.path.join(base_dir + str(random_state), 'val')
 
    # Load data using CustomMILDataGenerator
    train_generator = pc.CustomMILDataGenerator(train_dir, image_size, batch_size, is_training=True, num_workers=50)
    val_generator = pc.CustomMILDataGenerator(val_dir, image_size, batch_size, is_training=False, num_workers=50)
 
    # Compute class weights based on training data
    num_positive_bags, num_negative_bags = train_generator.num_positive_bags, train_generator.num_negative_bags
    total_bags = num_positive_bags + num_negative_bags
    class_weights = {
        0: total_bags / (2 * num_negative_bags),
        1: total_bags / (2 * num_positive_bags)
    }
    class_weights_dict = dict(enumerate(class_weights))
 
    # Iterate over different configurations (dropout, freeze layers, lambda values)
    for dropout_rate in dropout_rates:
        for freeze_layer in freeze_layers:
            for lambda_value in lambda_values:
                print(f"Training with random_state={random_state}, dropout_rate={dropout_rate}, freeze_layer={freeze_layer}, lambda_value={lambda_value}")
 
                # Wrap model creation inside the strategy scope for multi-GPU support
                with strategy.scope():
                    # Build the model using the current parameters
                    model = pc.MIL_Pos_MHA_Att(
                        input_shape=(None, *image_size, 3),
                        base_model_name='VGG16',  # Example base model
                        learning_rate=0.00001,
                        dropout_rate=dropout_rate,
                        freeze_layers=freeze_layer,
                        lambda_value=lambda_value
                    )
 
                    # Compile the model within the strategy scope
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
 
                # Callbacks for early stopping, learning rate reduction, and model checkpointing
                early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
 
                # Ensure save path includes random_state and current configuration
                model_save_path = os.path.join(
                    save_model_dir,
                    f"{save_model_name}_DP_{dropout_rate}_FL_{freeze_layer}_L_{lambda_value}_RS_{random_state}.keras"
                )
                model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
 
                # Train the model
                history = model.fit(
                    train_generator,
                    epochs=150,  # Define your number of epochs
                    validation_data=val_generator,
                    class_weight=class_weights_dict,  # class weights
                    callbacks=[early_stopping, reduce_lr, model_checkpoint]
                )
 