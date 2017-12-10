"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg

def reduce_dimensions(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """    
    #If performing dimensionality reduction on training data, create PCA vectors
    if 'fvectors_train' not in model:
        print("Training data")
        print("Create PCA Vectors and save first 10")
        covx = np.cov(feature_vectors_full, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
        v = np.fliplr(v)

        #Save training mean and pca vectors in model
        print("Store pca vectors and training mean in model")
        model['pca_vectors'] = v[:, 1:11].tolist()
        model['train_mean'] = np.mean(feature_vectors_full)
        print("PCA Vectors shape: " + str(np.array(model['pca_vectors']).shape))

        #Project training data onto pca vectors
        centred_train_data = feature_vectors_full - model['train_mean']
        return np.dot(centred_train_data, np.array(model['pca_vectors']))
    else:
        print('Testing data')

        #Project testing data onto pca vectors
        print('Projecting test data onto PCA Vectors')
        print(model['train_mean'])
        print(np.array(model['pca_vectors']).shape)
        centred_test_data = feature_vectors_full - model['train_mean']
        return np.dot(centred_test_data, np.array(model['pca_vectors']))

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    print ("Classifying " + str(len(page)) + " characters!")

    #Prepare training/test data and labels
    train = np.array(model['fvectors_train'])[:, :]
    labels_train = np.array(model['labels_train'])
    test = np.array(page)[:, :]
    
    print("Train shape: " + str(train.shape))
    print("Test shape: " + str(test.shape))
    print("Labels shape: " + str(labels_train.shape))
    
    print("Matrix multiply test data with training...")
    x = np.dot(test, train.transpose())
    print("Calc modtest and modtrain data...")
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))

    print("Calc distances between multiplied data and modtest and train...")
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    
    nearest = np.argmax(dist, axis=1)
    mdist = np.max(dist, axis=1)
    print("Get nearest neighbors")

    label = labels_train[nearest]

    #print(train)
    #print("Test")
    #print(test.shape)
    #print("Modtest")
    #print(modtest.shape)
    #print("Modtrain")
    #print(modtrain.shape)
    #print("Distances: ")
    #print(dist.shape)    
    #print("mdist")
    #print(mdist.shape)
    
    print(label)
    print("DONE")
    
    return label
    #Get nearest neighbor of each character
    #return np.repeat(labels_train[0], len(page))
