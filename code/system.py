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
import scipy.ndimage as ndimage

def create_pca_vectors(train_data, model):
    """Creates pca vectors from training data and stores in model

    Params:
    train_data - Train data vectors stored as rows
        in matrix
    model - a dictionary storing the outputs of the model
       training stage
    """    
    
    print("Create PCA Vectors and save first 10")
    covx = np.cov(train_data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
    v = np.fliplr(v)

    #Save training mean and pca vectors in model
    print("Store pca vectors and training mean in model")
    model['pca_vectors'] = v[:, 1:11].tolist()
    model['train_mean'] = np.mean(train_data)
    print("PCA Vectors shape: " + str(np.array(model['pca_vectors']).shape)) 

def reduce_dimensions(feature_vectors_full, model):
    """Takes feature vectors and projects them onto 10 pca vectors

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    #If performing dimensionality reduction on training data, create PCA vectors first
    if 'fvectors_train' not in model:
        create_pca_vectors(feature_vectors_full, model)

    #Project data onto pca vectors
    print('Project ' + str(feature_vectors_full.shape[0]) + ' feature vectors onto PCA Vectors')
    centred_data = feature_vectors_full - model['train_mean']
    return np.dot(centred_data, np.array(model['pca_vectors']))

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

    #Apply gaussian blur to image
    #Couldnt get blur to work
    #images_test = blur_images(images_test)

    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def blur_images(images):
    #Estimate blur

    #Create list of blurred images
    #return [ndimage.filters.gaussian_filter(img, 0.4) for img in images]
    print("Type of list before: " + str(type(images)))
    print("Type of list after: " + str(type([cv2.bilateralFilter(img,9,75,75) for img in images]))) 
    print("Type of image: ")
    print(images[0].shape)
    print(cv2.bilateralFilter(images[0], 9, 75, 75).shape)
    return [cv2.bilateralFilter(img,9,75,75) for img in images]

def classify_page(page, model):
    """KNN classifier. Returns most common label among k neighbors

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    print ("Classifying " + str(len(page)) + " characters!")

    #Prepare training/test data and labels
    train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    test = np.array(page)
    
    #Calculate cos distance for each test point to each train point
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance

    #Calculate k nearest algorithm
    k = calc_good_k(dist)

    #Get nearest labels and weight them
    nearestK_dist = np.fliplr(np.sort(dist, axis=1))[:, :k]
    nearest_k_dist_indices = np.fliplr(np.argsort(dist, axis=1))[:, :k]
    #nearestK_dist = dist[nearest_k_dist_indices]
    nearestk_labels = labels_train[nearest_k_dist_indices]
    #nearest_weighted_labels = np.stack((nearestk_labels, nearestK_dist), axis = 2)

    #Fill list with (uniqueLabel, count of each label, and indices for original labels) for each test sample
    #Replace with unique label with max count
    unique_label_counts = [np.unique(l, return_counts = True) for l in nearestk_labels]
    #List of arrays of unique chars
    unique_labels = [np.unique(l) for l in nearestk_labels]
    #print(unique_labels)

    #Make list of list of total weights
    #unique_neighbor_highscore = [[np.sum(nearestK_dist[i, (nearestk_labels[i] == l)]) for l in c] for (i, c) in enumerate(unique_labels)]
    #unique_neighbor_highscore_label = [unique_labels[i][np.argmax(np.array(s))] for (i, s) in enumerate(unique_neighbor_highscore)]
    #print(unique_label_scores)
    most_common_neighbors = [u[np.argmax(c)] for (u,c) in unique_label_counts]

    return np.array(most_common_neighbors)

def calc_good_k(dist_mat):
    return 250

def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    print(page.shape)

    return labels