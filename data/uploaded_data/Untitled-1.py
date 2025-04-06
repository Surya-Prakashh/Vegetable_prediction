    # %%
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import *
    from keras.models import *
    from keras.preprocessing import image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    import os, shutil
    import warnings
    warnings.filterwarnings('ignore')

    # %% [markdown]
    # # Visualise the Images

    # %%
    train_path = r"D:\Sem6\Deep Learning\ex9\Vegetable Images\train"
    validation_path = r"D:\Sem6\Deep Learning\ex9\Vegetable Images\validation"
    test_path = r"D:\Sem6\Deep Learning\ex9\Vegetable Images\test"

    # Get list of categories (subfolders in the train directory)
    image_categories = os.listdir(train_path)

    # Function to plot images
    def plot_images(image_categories):
        plt.figure(figsize=(12, 12))  # Set figure size
        
        for i, cat in enumerate(image_categories[:16]):  # Limit to 16 categories
            image_path = os.path.join(train_path, cat)  # Correct path joining
            images_in_folder = os.listdir(image_path)
            
            if not images_in_folder:
                continue  # Skip empty folders
            
            first_image_path = os.path.join(image_path, images_in_folder[0])
            img = image.load_img(first_image_path)  # Load image
            img_arr = image.img_to_array(img) / 255.0  # Convert to array & normalize
            
            plt.subplot(4, 4, i+1)  # Create subplot grid (4x4)
            plt.imshow(img_arr)
            plt.title(cat)
            plt.axis('off')  # Hide axes
        
        plt.show()

    # Call the function to plot images
    plot_images(image_categories)

    # %% [markdown]
    # # Prepare the Dataset

    # %%
    # Creating Image Data Generator for train, validation and test set

    # 1. Train Set
    train_gen = ImageDataGenerator(rescale = 1.0/255.0) # Normalise the data
    train_image_generator = train_gen.flow_from_directory(
                                                train_path,
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='categorical')

    # 2. Validation Set
    val_gen = ImageDataGenerator(rescale = 1.0/255.0) # Normalise the data
    val_image_generator = train_gen.flow_from_directory(
                                                validation_path,
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='categorical')

    # 3. Test Set
    test_gen = ImageDataGenerator(rescale = 1.0/255.0) # Normalise the data
    test_image_generator = train_gen.flow_from_directory(
                                                test_path,
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='categorical')

    # %%
    # Print the class encodings done by the generators
    class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])
    print(class_map)

    # %% [markdown]
    # # Building a CNN model

    # %%
    # Build a custom sequential CNN model

    model = Sequential() # model object

    # Add Layers
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[150, 150, 3]))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(2))

    # Flatten the feature map
    model.add(Flatten())

    # Add the fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(15, activation='softmax'))

    # print the model summary
    model.summary()

    # %%
    # Compile and fit the model
    early_stopping = keras.callbacks.EarlyStopping(patience=5)  # ✅ Correct

    model.compile(optimizer='Adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])  # ✅ Correct (Use list format for metrics)

    hist = model.fit(train_image_generator, 
                    epochs=14, 
                    verbose=1, 
                    validation_data=val_image_generator, 
                    steps_per_epoch=15000//32, 
                    validation_steps=3000//32, 
                    callbacks=[early_stopping])  # ✅ Correct (Wrap in a list)


    # %% [markdown]
    # **Model trained for 15 Epochs**

    # %%
    # Plot the error and accuracy
    h = hist.history
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(h['loss'], c='red', label='Training Loss')
    plt.plot(h['val_loss'], c='red', linestyle='--', label='Validation Loss')
    plt.plot(h['accuracy'], c='blue', label='Training Accuracy')
    plt.plot(h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
    plt.xlabel("Number of Epochs")
    plt.legend(loc='best')
    plt.show()

    # %%
    # Predict the accuracy for the test set
    model.evaluate(test_image_generator)

    # %%
    # Testing the Model
    test_image_path = r'D:\Sem6\Deep Learning\ex9\Vegetable Images\test\Broccoli\1002.jpg'

    def generate_predictions(test_image_path, actual_label):
        
        # 1. Load and preprocess the image
        test_img = image.load_img(test_image_path, target_size=(150, 150))
        test_img_arr = image.img_to_array(test_img)/255.0
        test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))

        # 2. Make Predictions
        predicted_label = np.argmax(model.predict(test_img_input))
        predicted_vegetable = class_map[predicted_label]
        plt.figure(figsize=(4, 4))
        plt.imshow(test_img_arr)
        plt.title("Predicted Label: {}, Actual Label: {}".format(predicted_vegetable, actual_label))
        plt.grid()
        plt.axis('off')
        plt.show()

    # call the function
    generate_predictions(test_image_path, actual_label='Broccoli')

    # %%
    # Generate predictions for external image
    external_image_path_2 = r"C:\\Users\Surya_prakash\Downloads\\Untitled design.png"
    generate_predictions(external_image_path_2, actual_label='Potato')

    # %%
    # Save the model in HDF5 format (.h5)
    model.save("vegetable_classifier.keras")

    # Save the model in TensorFlow's SavedModel format



    # %%



