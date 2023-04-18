import os
import random
import sys
import cv2
import imageio.v2 as imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adadelta, Adam
from keras_preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tqdm import tqdm
from keras.initializers import GlorotNormal
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from pathlib import Path

# Set a random seed for reproducibility
random.seed(42)

# Set the path to the InterdisciplinaryDataAnalysisClass_FinalDataSet folder on your local machine
data_folder_path = '/Users/jay/InterdisciplinaryDataAnalysisClass_FinalDataSet'

# Construct the file path for the metadata file
data_folder = Path(data_folder_path)
metadata_file_path = data_folder / 'sampleMetadata_concatenated.txt'

# Read the metadata file as a pandas DataFrame and define folder_names for image folders
metaData = pd.read_csv(metadata_file_path, sep='\t', dtype={'Timestamp': str})
folder_names = ['Local', 'Overview', 'Window', 'Pixel']

# Create a dictionary to store image arrays for 'Local', 'Overview', and 'Window' folders
image_arrays = {folder_name.lower() + '_images': [] for folder_name in ['Local', 'Overview', 'Window']}

# Initialize a StandardScaler object for normalizing the images before training
scaler = StandardScaler()


# Randomly select a specified number of samples from the metadata
num_samples = 48000
indices = random.sample(range(metaData.shape[0]), num_samples)

# Create a new DataFrame with only the selected entries
metaData_sample = metaData.iloc[indices].reset_index(drop=True)

# Add new columns to store the extracted RGB values
metaData_sample['R'] = 0
metaData_sample['G'] = 0
metaData_sample['B'] = 0

# Loop through the selected indices and process the images
for i in tqdm(indices):
    # Loop through the folder names (Local, Overview, Window, Pixel)
    for folder_name in folder_names:
        # Construct the image file path using the data folder path, folder name, sample number, and timestamp
        img_path = f'{data_folder_path}/{folder_name}/{folder_name.lower()}.{metaData["Sample Number"][i]}.{metaData["Timestamp"][i]}.png'
        
        # Check if the image file exists
        if os.path.exists(img_path):
            # Read the image using OpenCV and convert from BGR to RGB color space
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize the image for 'Overview' and 'Montage' folders
            if folder_name in ['Overview', 'Montage']:
                img = cv2.resize(img, (100, 100))

            # For 'Pixel' folder, extract RGB values and store them in the metadata dataframe
            if folder_name == 'Pixel':
                center_rgb = img[0, 0] / 255
                metaData_sample.loc[metaData_sample['Sample Number'] == metaData["Sample Number"][i], ['R', 'G', 'B']] = center_rgb
            else:
                # Reshape the image into a 2D array for scaling
                img_2d = img.reshape(-1, img.shape[-1])
                
                # Scale the image using StandardScaler
                img_scaled = scaler.fit_transform(img_2d)
                
                # Reshape the scaled image back to its original shape
                img = img_scaled.reshape(img.shape)
                
                # Append the processed image to the corresponding image_arrays entry
                image_arrays[folder_name.lower() + '_images'].append(img)



# Check the number of images loaded in each folder
folder_names = ['Local', 'Overview', 'Window']
for folder_name in folder_names:
    print(f"Number of {folder_name.lower()} images: {len(image_arrays[folder_name.lower() + '_images'])}")

# Display a random image from each folder
for folder_name in folder_names:
    random_index = np.random.randint(len(image_arrays[folder_name.lower() + '_images']))
    img = image_arrays[folder_name.lower() + '_images'][random_index]
    print(f"{folder_name} image shape: {img.shape}")
    plt.imshow(img)
    plt.title(f"{folder_name} image")
    plt.show()



# Categorical columns of the data frame, ordered by the number of occurences of each class
categorical_columns = ['HerbariumSheet', 'Plant', 'Inflorescence', 'External', 'Text', 'FlowerFruit', 
                       'Vegetative', 'Typed', 'Leaf', 'Border', 'DataLabel', 'BorderDifferentComponents', 
                       'ColorBarRuler', 'DeterminationSlip', 'Stem', 'Lamina', 'Background', 'Barcode', 
                       'ColorBoxMatrix', 'InstitutionalInsigniaIcon', 'Handwritten', 'Envelope', 
                       'AccessionInformation', 'BarsDemarcationLines', 'Damage', 'TapeString', 'Glue', 
                       'Petiole', 'Tendril', 'BorderSameComponents', 'HerbariumSheetMargin', 'Other']

# Specify the number of classes that you want the model to consider and extract them to a list
num_classes = 16 
top_classes = categorical_columns[:num_classes]

# Extract the numerical data from the specified columns in the metaData_sample DataFrame
mlp_input = metaData_sample[['R', 'G', 'B']].values

# Convert the list of 'local_images' to a NumPy array
local_images = np.array(image_arrays['local_images'])

# Convert the list of 'overview_images' to a NumPy array
overview_images = np.array(image_arrays['overview_images'])

# Convert the list of 'window_images' to a NumPy array
window_images = np.array(image_arrays['window_images'])



# Extract the target values (y) from the metaData_sample DataFrame
y = metaData_sample[top_classes].values

# Set the split ratio for training and validation sets
split_ratio = 0.8

# Split the data into training and validation sets using the train_test_split function
(
    local_train, local_val, 
    overview_train, overview_val, 
    window_train, window_val, 
    mlp_train, mlp_val, 
    y_train, y_val
) = train_test_split(
    local_images, overview_images, window_images, mlp_input, y,
    train_size=split_ratio, random_state=42
)



# Define the window branch of the CNN
def create_window_branch(input_shape):
    input_layer = Input(shape=input_shape, name="window_input")
    x = Conv2D(32, kernel_size=(3, 3), kernel_initializer=GlorotNormal(), activation='relu', name="window_conv1")(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), name="window_maxpool1")(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=(3, 3), kernel_initializer=GlorotNormal(), activation='relu', name="window_conv2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), name="window_maxpool2")(x)
    x = Dropout(0.25)(x)
    x = Flatten(name="window_flatten")(x)
    return input_layer, x

# Define the overview branch of the CNN
def create_overview_branch(input_shape):
    input_layer = Input(shape=input_shape, name="overview_input")
    x = Conv2D(32, kernel_size=(3, 3), kernel_initializer=GlorotNormal(), activation='relu', name="overview_conv1")(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name="overview_maxpool1")(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=(2, 2), kernel_initializer=GlorotNormal(), activation='relu', name="overview_conv2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name="overview_maxpool2")(x)
    x = Dropout(0.25)(x)
    x = Flatten(name="overview_flatten")(x)
    return input_layer, x

# Define the local branch of the CNN
def create_local_branch(input_shape):
    input_layer = Input(shape=input_shape, name="local_input")
    x = Conv2D(16, kernel_size=(3, 3), kernel_initializer=GlorotNormal(), activation = 'relu', name = "local_conv1")(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name="local_maxpool1")(x)
    x = Dropout(0.25)(x)
    x = Flatten(name="local_flatten")(x)
    return input_layer, x

# Define the MLP for categorical data
def create_mlp_branch(input_shape, name):
    input_layer = Input(shape=input_shape, name=f"{name}_input")
    x = Dense(128, activation='relu', name=f"{name}_dense1")(input_layer)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu', name=f"{name}_dense2")(x)
    return input_layer, x




# Create CNN branches for each image type
local_input, local_branch = create_local_branch(input_shape=(7, 7, 3))
overview_input, overview_branch = create_overview_branch(input_shape=(100, 100, 3))
window_input, window_branch = create_window_branch(input_shape=(31, 31, 3))

# Create MLP branch 
categorical_input, categorical_branch = create_mlp_branch(input_shape=(3,), name='MLP')

# Concatenate the outputs of each branch
concatenated = Concatenate()([local_branch, overview_branch, window_branch, categorical_branch])

# Add a fully connected layer
fc = Dense(128, activation='relu')(concatenated)
fc = Dropout(0.25)(fc) 
fc = Dense(64, activation='relu')(fc)
fc = Dropout(0.25)(fc) 
fc = Dense(32, activation='relu')(fc)
fc = Dropout(0.25)(fc) 
fc = Dense(32, activation='relu')(fc)
categorical_output = Dense(num_classes, activation='sigmoid', name='categorical_output')(fc)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)

# Create and compile the model
model = Model(inputs=[local_input, overview_input, window_input, categorical_input], outputs=[categorical_output])
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])

# Display the model summary
model.summary()



# Converges in ~3 epochs

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_modelmini.h5', save_best_only=True, monitor='val_categorical_accuracy', mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=3, min_lr=1e-10, verbose=1)

# Train the model
batch_size = 128
epochs = 10
history = model.fit(
    [local_train, overview_train, window_train, mlp_train],
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([local_val, overview_val, window_val, mlp_val], y_val),
    callbacks=[lr_scheduler, checkpoint]
)

# Save the model
model.save('/Users/Jay/InterdisciplinaryDataAnalysisClass_FinalDataSet/')







