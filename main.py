### LIBRARY IMPORTS ###
import os
import numpy as np

import keras
import keras.applications as ka
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.utils import img_to_array

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def load_data(path:str)->np.ndarray:
    '''
    Load dataset from path as numpy array of PIL images.

    Input:
        - path: A string of the path to the dataset directory

    Output:
        - dataset: np.ndarray with shape (1000, 2). 
            - dataset(:, 0): Image data loaded as np.ndarray
            - dataset(:, 1): Corresponding classnames inferred from data's directory
    '''

    from collections import defaultdict
    class_counts = defaultdict(int)

    def load_img(dir, filename, class_label):
        class_counts[class_label] += 1
        return keras.utils.load_img(os.path.join(dir, filename)), class_label # keras loads as PIL image

    class_names = os.listdir(path)
    sub_dirs    = [os.path.join(path, class_name) for class_name in class_names] 

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Loop class directories then images, loading data with their class name.
    dataset = [
               (load_img(sub_dir, filename, class_label))
               for sub_dir, class_label in zip(sub_dirs, class_names)
               for filename in os.listdir(sub_dir)
               if os.path.splitext(filename)[-1] in valid_extensions
               ]

    # Print out class counts
    for class_label, count in class_counts.items():
        print(f"{class_label}:{count} images")

    return np.array(dataset, dtype=object)

def split_data(X:np.ndarray, Y:np.ndarray, train_fraction:float, randomize=False, eval_set=True)-> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().
    
    Insert a more detailed description here.

    Split the dataset (X, Y) into train, test, and optionally evaluation sets per class.
    
    Parameters:
    - X: numpy array of image data.
    - Y: numpy array of labels (must match X in length).
    - train_fraction: proportion of non-eval data used for training (0 < train_fraction < 1).
    - randomize: if True, shuffles data within each class before splitting.
    - eval_set: if True, returns a third evaluation set of fixed size (default: True).
    
    Insert a more detailed description here.

    NOTE: Soft copying the dataset.

    Input:
        - X: numpy array of training data to be split
        - Y: numpy array of class names corresponding to X
        - train_fraction: The fraction of the dataset that should be in the train dataset
        - randomize=False: When True the data is shuffled before returned
        - eval_set=True: When True function return an evaluation dataset

    Output: 
        - train_dataset: The dataset to be used for training
        - test_dataset: The dataset to be used for testing
        
        When eval_set==True
        - eval_dataset: The dataset to be used for evalution
    """

    dataset = np.stack((X.copy(), Y.copy()), axis=1)

    dataset[dataset[:, 1].argsort(kind='stable')] # Sort by class for ease of segmentation

    values, class_lengths = np.unique(dataset[:,1], return_counts=True)
    num_of_classes = len(values)

    # Shuffle images within classes
    if(randomize):
        np.random.shuffle(dataset[0:class_lengths[1]])
        for i in range(0, num_of_classes-1):
            np.random.shuffle(dataset[class_lengths[i]:class_lengths[i+1]])

    train_sizes = (class_lengths*train_fraction).astype(int)
    test_sizes  = class_lengths - train_sizes
    if eval_set:
        test_sizes  = (test_sizes/2).astype(int)
    eval_sizes  = class_lengths - train_sizes - test_sizes

    ordered_sizes = np.vstack((train_sizes, test_sizes, eval_sizes)).flatten('F')
    positions = np.cumsum(ordered_sizes)

    split_datasets = np.split(dataset, positions[:-1]) # don't split at len(dataset) (i.e. positions[-1])

    train_dataset = np.concatenate(split_datasets[0::3])
    test_dataset  = np.concatenate(split_datasets[1::3])
    eval_dataset  = np.concatenate(split_datasets[2::3])

    # train_dataset, test_dataset, eval_dataset = np.concatenate(split_datasets[np.r_[0::3, 1::3, 2::3]], axis=1) # Not as good way to do it IMO

    # Shuffle between classes
    if(randomize):
        np.random.shuffle(train_dataset)
        np.random.shuffle(test_dataset)
        np.random.shuffle(eval_dataset)

    if eval_set:
        return train_dataset, test_dataset, eval_dataset
    return train_dataset, test_dataset

def load_model()-> keras.Model:
    '''
    Load in a model using the tf.keras.applications model and return it.
    
    Model: MobileNetV2
    '''

    #-----Defaults-------#
    # input_shape=None,
    # alpha=1.0,
    # include_top=True,
    # weights="imagenet",
    # input_tensor=None,
    # pooling=None,
    # classes=1000,
    # classifier_activation="softmax",   
    #--------------------#

    model = ka.MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=(224, 224, 3))

    return model

def load_model_dense_layer():
    '''
    Load and modify the pre-trained MobileNetV2 model for 5-class flower classification.
    
    Returns:
        modified_model: The modified MobileNetV2 model with the new Dense layer.
    '''
    # Load MobileNetV2 without the top layer

    base_model = load_model()

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    #keep first 2 top layers open

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    predictions = Dense(5, activation='softmax')(x)
    modified_model = Model(inputs=base_model.input, outputs=predictions)

    return modified_model


def transfer_learning(train_set, eval_set, model: keras.Model, parameters):
    '''
    Implement and perform standard transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)

    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2

    '''

    learning_rate, momentum, nesterov = parameters

    # Compile the model with the given optimizer settings
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=4,
                               min_delta=0.001, # minimum change to be considered improvement
                               restore_best_weights=True, 
                               verbose=1)

    # Train the model
    history = model.fit(*train_set,
                        validation_data=eval_set, 
                        epochs=21, 
                        batch_size=32, 
                        verbose=1, 
                        callbacks=[early_stop])

    return model, history


def preprocess_data(dataset):
    '''
    Resize images to (224, 224, 3), normalize pixel values, and convert class labels to integers.
    
    Parameters:
        dataset: iterable of (image, label) pairs
    
    Returns:
        processed dataset: Tuple of numpy arrays ready for model input
    '''

    # Mapping from class name to index
    class_name_to_index = {
        'daisy'     : 0,
        'dandelion' : 1,
        'roses'     : 2,
        'sunflowers': 3,
        'tulips'    : 4
    }

    def pre_process(data):
        assert len(data) == 2, "Invalid data format"
        
        image, label = data

        label_index = class_name_to_index.get(label, -1)
        assert label_index != -1, f"Unknown label: {label}"

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        image = image.resize((224, 224))
        image = img_to_array(image, dtype=np.float32) / 255.0  # Normalize image

        return image, label_index

    processed_dataset = map(pre_process, dataset) 

    return processed_dataset

#Early stopping prevents overfitting and saves time  
def plot_training_history(history):
    # Retrieve metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def k_fold_validation(features: np.ndarray, ground_truth:np.ndarray, classifier: keras.Model, k=3):
    '''
    Inputs:
        - features: np.ndarray of features in the dataset
        - ground_truth: np.ndarray of class values associated with the features
        - fit_func: f
        - classifier: class object with both fit() and predict() methods which
        can be applied to subsets of the features and ground_truth inputs.
        - predict_func: function, calling predict_func(features) should return
        a numpy array of class predictions which can in turn be input to the 
        functions in this script to calculate performance metrics.
        - k: int, number of sub-sets to partition the data into. default is k=2
    Outputs:
        - avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
        The first row is the average precision for each class over the k
        validation steps. Second row is recall and third row is f1 score.
        - sigma_metrics: np.ndarray, each value is the standard deviation of 
        the performance metrics [precision, recall, f1_score]
    '''

    dataset = np.stack((features.copy(), ground_truth.copy()), axis=1)

    # Split data
    partitions = []
    for i in range(k, 0, -1):
        partition, dataset = split_data(dataset, None, 1/i, randomize=True, eval_set=False) # type: ignore
        partition = preprocess_data(partition)
        partitions.append(partition)

    num_of_classes = len(np.unique(ground_truth))

    precisions = np.empty((k, num_of_classes))
    recalls = np.empty((k, num_of_classes))
    f1s = np.empty((k, num_of_classes))

    #go through each partition and use it as a test set.
    for partition_no in range(k):

        #determine test and train sets
        train_set = partitions[:partition_no] + partitions[partition_no+1:] # Remove 'partition_no' 
        train_set = [item for partition in train_set for item in partition] # Merge remaining partitions
        test_set  = partitions[partition_no]

        temp_classifier = keras.models.clone_model(classifier)

        temp_classifier.compile(optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

        temp_classifier.set_weights(classifier.get_weights())

        #fit model to training data and perform predictions on the test set
        temp_classifier.fit(train_set[0], train_set[1],
                        epochs=15)

        predictions = temp_classifier.predict(test_set[0])
        pred_classes = np.argmax(predictions, axis=1)

        #calculate performance metrics
        precisions[partition_no] = precision_score(test_set[1], pred_classes, average=None)
        recalls[partition_no] = recall_score(test_set[1], pred_classes, average=None)
        f1s[partition_no] = f1_score(test_set[1], pred_classes, average=None)

    #perform statistical analyses on metrics
    avg_precision = np.average(precisions, axis=0)
    avg_recall = np.average(recalls, axis = 0)
    avg_f1 = np.average(f1s, axis = 0)

    avg_metrics = np.array([avg_precision, avg_recall, avg_f1])

    std_precision = np.std(precisions, axis=0)
    std_recall = np.std(recalls, axis = 0)
    std_f1 = np.std(f1s, axis = 0)

    sigma_metrics = np.array([std_precision, std_recall, std_f1])
    
    return avg_metrics, sigma_metrics

#should be same performance but a 1/10 of the time
def accelerated_learning(x_train_features, y_train, 
                         x_val_features, y_val, 
                         parameters):
    '''
    Perform accelerated learning using precomputed features from a frozen base model.

    Parameters:
    - x_train_features: Features extracted from frozen base model (train set)
    - y_train: Training labels
    - x_val_features: Features extracted from frozen base model (validation set)
    - y_val: Validation labels
    - parameters: Tuple (learning_rate, momentum, nesterov)

    Returns:
    - model: Trained top classifier model
    - history: Training history
    '''

    learning_rate, momentum, nesterov = parameters

    # Define the custom top layers (classifier)
    model = Sequential([
        GlobalMaxPooling2D(input_shape=x_train_features.shape[1:]),
        Dense(5, activation='softmax')  # Assuming 5 classes
    ])

    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        min_delta=0.001,  # minimum change to be considered improvement
        restore_best_weights=True,
        verbose=1
    )

    # Train the classifier
    history = model.fit(
        x_train_features, y_train,
        validation_data=(x_val_features, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history

def compare_histories(history_standard, history_accelerated):
    epochs_standard = range(1, len(history_standard.history['accuracy']) + 1)
    epochs_accelerated = range(1, len(history_accelerated.history['accuracy']) + 1)

    plt.figure(figsize=(14, 6))

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_standard, history_standard.history['val_accuracy'], 'b-', label='Standard - Val Accuracy')
    plt.plot(epochs_accelerated, history_accelerated.history['val_accuracy'], 'g-', label='Accelerated - Val Accuracy')
    plt.plot(epochs_standard, history_standard.history['accuracy'], 'b--', label='Standard - Train Accuracy')
    plt.plot(epochs_accelerated, history_accelerated.history['accuracy'], 'g--', label='Accelerated - Train Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # --- Loss Plot ---
    plt.subplot(1, 2, 2)
    
    plt.plot(epochs_standard, history_standard.history['val_loss'], 'r-', label='Standard - Val Loss')
    plt.plot(epochs_accelerated, history_accelerated.history['val_loss'], 'orange', label='Accelerated - Val Loss')
    plt.plot(epochs_standard, history_standard.history['loss'], 'r--', label='Standard - Train Loss')
    plt.plot(epochs_accelerated, history_accelerated.history['loss'], 'orange', linestyle='--', label='Accelerated - Train Loss')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataset = load_data('small_flower_dataset')
    split_data(dataset[:, 0], dataset[:, 1], 0.8, eval_set=False)


# train_set, test_set, eval_set = split_data(dataset[:, 0], dataset[:, 1], 0.8, randomize=True, eval_set=True)

# modified_model = load_model_dense_layer()

# # Preprocess train and evaluation sets
# train_set = preprocess_data(train_set)
# eval_set = preprocess_data(eval_set)

# # Train the model
# parameters = (0.01, 0.0, False)
# trained_model, history = transfer_learning(train_set, eval_set, modified_model, parameters)
# trained_model.evaluate(eval_set[0], eval_set[1]) 

# history_standard = history

# # Call the function
# plot_training_history(history)

# def plot_comparison(histories, metric='accuracy'):
#     plt.figure(figsize=(10, 6))
    
#     for lr, history in histories.items():
#         if metric == 'accuracy':
#             plt.plot(history.history[metric], label=f'Train Acc (lr={lr})')
#             plt.plot(history.history[f'val_{metric}'], label=f'Val Acc (lr={lr})', linestyle='--')
#         elif metric == 'loss':
#             plt.plot(history.history[metric], label=f'Train Loss (lr={lr})')
#             plt.plot(history.history[f'val_{metric}'], label=f'Val Loss (lr={lr})', linestyle='--')
#         else:
#             plt.plot(history.history[metric], label=f'Train {metric.title()} (lr={lr})')
#             plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.title()} (lr={lr})', linestyle='--')
    
#     plt.title(f'Training & Validation {metric.title()} for Different Learning Rates')
#     plt.xlabel('Epochs')
#     plt.ylabel(metric.title())
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# learning_rates = [0.1, 0.01, 0.001, 0.015] #0.015 had the most optimal performance

# histories = {}
# models = {}

# for lr in learning_rates:
#     print(f"\nTraining with learning rate = {lr}")
#     model = load_model_dense_layer()  # reload model each time to reset weights
#     parameters = (lr, 0.0, False)
#     trained_model, history = transfer_learning(train_set, eval_set, model, parameters)
#     models[lr] = trained_model
#     histories[lr] = history

# plot_comparison(histories, metric='accuracy')
# plot_comparison(histories, metric='loss')


# best_model = models[0.015]

# # Predict on test data
# x_test, y_test = preprocess_data(test_set)
# y_pred = best_model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Automatically detect labels that are present
# unique_labels = np.unique(np.concatenate((y_test, y_pred_classes)))
# class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
# label_names = [class_names[i] for i in unique_labels]

# # Compute and plot confusion matrix
# cm = confusion_matrix(y_test, y_pred_classes, labels=unique_labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
# disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# plt.title("Confusion Matrix - Test Dataset")
# plt.tight_layout()
# plt.show()
 
# ## Your code
# precision = precision_score(y_test, y_pred_classes, average=None)
# recall = recall_score(y_test, y_pred_classes, average=None)
# f1 = f1_score(y_test, y_pred_classes, average=None)

# print("precision: ", precision)
# print("recall: ", recall)
# print("f1: ", f1)

# dataset = load_data('small_flower_dataset')

# model = load_model_dense_layer()

# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])

# avg_metrics, sigma_metrics = k_fold_validation(dataset[:, 0], dataset[:, 1], model, k=3)

# precision_1 = avg_metrics[0]
# recall_1 = avg_metrics[1]
# f1_1 = avg_metrics[2]

# std_precision_1 = sigma_metrics[0]
# std_recall_1 = sigma_metrics[1]
# std_f1_1 = sigma_metrics[2]

# avg_metrics, sigma_metrics = k_fold_validation(dataset[:, 0], dataset[:, 1], model, k=5)

# precision_2 = avg_metrics[0]
# recall_2 = avg_metrics[1]
# f1_2 = avg_metrics[2]

# std_precision_2 = sigma_metrics[0]
# std_recall_2 = sigma_metrics[1]
# std_f1_2 = sigma_metrics[2]

# avg_metrics, sigma_metrics = k_fold_validation(dataset[:, 0], dataset[:, 1], model, k=10)

# precision_3 = avg_metrics[0]
# recall_3 = avg_metrics[1]
# f1_3 = avg_metrics[2]

# std_precision_3 = sigma_metrics[0]
# std_recall_3 = sigma_metrics[1]
# std_f1_3 = sigma_metrics[2]

# print(f"Average precision (K = 2): {precision_1}")
# print(f"Average recall (K = 2): {recall_1}")
# print(f"Average f1 (K = 2): {f1_1}")

# print(f"Precision sigma (K = 2): {std_precision_1}")
# print(f"Recall sigma (K = 2): {std_recall_1}")
# print(f"F1 sigma (K = 2): {std_f1_1}")

# print(f"Average precision (K = 3): {precision_2}")
# print(f"Average recall (K = 3): {recall_2}")
# print(f"Average f1 (K = 3): {f1_2}")

# print(f"Precision sigma (K = 3): {std_precision_2}")
# print(f"Recall sigma (K = 3): {std_recall_2}")
# print(f"F1 sigma (K = 3): {std_f1_2}")

# print(f"Average precision (K = 5): {precision_3}")
# print(f"Average recall (K = 5): {recall_3}")
# print(f"Average f1 (K = 5): {f1_3}")

# print(f"Precision sigma (K = 5): {std_precision_3}")
# print(f"Recall sigma (K = 5): {std_recall_3}")
# print(f"F1 sigma (K = 5): {std_f1_3}")


# classes = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# # Bar width and positions
# bar_width = 0.25
# x = np.arange(len(classes))

# # Data with stds
# data = [
#     (precision_1, precision_2, precision_3, std_precision_1, std_precision_2, std_precision_3),
#     (recall_1, recall_2, recall_3, std_recall_1, std_recall_2, std_recall_3),
#     (f1_1, f1_2, f1_3, std_f1_1, std_f1_2, std_f1_3),
# ]

# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# metrics = ['Precision', 'Recall', 'F1 Score']

# # Plot with error bars
# for i, ax in enumerate(axs):
#     d1, d2, d3, std1, std2, std3 = data[i]
#     ax.bar(x - bar_width, d1, yerr=std1, width=bar_width, label='K=2', capsize=5)
#     ax.bar(x, d2, yerr=std2, width=bar_width, label='K=3', capsize=5)
#     ax.bar(x + bar_width, d3, yerr=std3, width=bar_width, label='K=5', capsize=5)
#     ax.set_title(f'{metrics[i]} (with Std Dev)')
#     ax.set_xticks(x)
#     ax.set_xticklabels(classes)
#     ax.set_ylim(0.3, 0.95)
#     ax.legend()
#     ax.grid(axis='y')

# plt.tight_layout()
# plt.show()

# momentum_values = [0.003, 0.06, 0.9]

# best_learning_rate = 0.015

# momentum_histories = {}
# momentum_models = {}

# for m in momentum_values:
#     print(f"\nTraining with learning rate = {best_learning_rate}, momentum = {m}")
#     model = load_model_dense_layer()  # Reload base model to reset weights
#     parameters = (best_learning_rate, m, False)  # Nesterov = False
#     trained_model, history = transfer_learning(train_set, eval_set, model, parameters)
#     momentum_models[m] = trained_model
#     momentum_histories[m] = history

# def plot_momentum_comparison(histories, metric='accuracy'):
#     plt.figure(figsize=(10, 6))
#     for momentum, history in histories.items():
#         plt.plot(history.history[metric], label=f'Train acc (momentum={momentum})')
#         plt.plot(history.history[f'val_{metric}'], label=f'Val acc (momentum={momentum})', linestyle='--')
#     plt.title(f'Training & Validation {metric.title()} for Different Momentum Values (lr=0.01)')
#     plt.xlabel('Epochs')
#     plt.ylabel(metric.title())
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Accuracy
# plot_momentum_comparison(momentum_histories, metric='accuracy')

# # Loss
# plot_momentum_comparison(momentum_histories, metric='loss')

# # Load base model with frozen weights
# base_model = ka.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False

# # Create a feature extractor model
# feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# x_train, y_train = train_set
# x_val, y_val = eval_set


# x_train_features = feature_extractor.predict(x_train, batch_size=32)
# x_val_features = feature_extractor.predict(x_val, batch_size=32)

# parameters = (0.01, 0.0, False)  # Example learning rate setup
# model_accelerated, history_accelerated = accelerated_learning(
#     x_train_features, y_train,
#     x_val_features, y_val,
#     parameters
# )

# #Aim: reduce the time and results similar to task 7
# compare_histories(history_standard, history_accelerated)