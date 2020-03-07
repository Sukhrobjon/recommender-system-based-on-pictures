"""
Class for 2000
"""

from keras.preprocessing.image import load_img
# from PIL import Image, load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import pandas as pd
import csv
import os
# load the model
model = VGG16(weights='imagenet', include_top=False)


class ImageRecommender():
    """
    Class makes a recommendation for image
    """
    def __init__(self, filepath):
        """
        Initialize the image recommender
        """
        self.model = model
        self.filepath = filepath
    
    def preprocess_image_for_vgg(self, filepath, img_id):
        """
            Takes an image and returns 4D array of features from vgg16 model
            
            Args:
                image_path: path to image
        """
        # load an image from file
        original_img = load_img(filepath + img_id, target_size=(224, 224))
        # convert the image pixels to a numpy array
        nparry_img = img_to_array(original_img)
        # # prepare the image for the VGG model
        image = np.expand_dims(nparry_img, axis=0)
        # print("image shape:", image.shape)
        processed_image = preprocess_input(image)
        return processed_image

    def get_feature_vector_from_vgg(self, processed_image):
        # dont use include top for vgg, because we dont want mlp part
        model = VGG16(weights='imagenet', include_top=False)

        feature_vector = model.predict(processed_image)
        return feature_vector


    def get_all_image_ids(filepath):
        images_ids = []
        # listing all the image names into list they are ids of the image
        for img in os.listdir(filepath):
            if img.endswith('jpg'):
                images_ids.append(img)

        return images_ids

# reshape the feature_vector so we can find cosine similarity score


def reshape_feature_vector(feature_vector):
    """
    Takes 4d array, return 2D representation
    """
    # this is the vgg output 7, 7, 512
    return feature_vector.reshape(1, 7*7*512)

# print(reshape_feature_vector(img_features))


def list_of_vgg_features(all_imgs):
    # creates the list of vgg features and reshape it ready for making cosine sim score later
    all_feature_vectors = []

    for index, img in enumerate(all_imgs):
        feature_vector = get_feature_vector_from_vgg(img)
        # reshape the feature vector
        feature_vector = reshape_feature_vector(feature_vector)

        all_feature_vectors.append(feature_vector)

    # return all feature vectors
    return all_imgs, all_feature_vectors


def create_cosine_table(list_of_ids, list_of_features):
    """Creates a cosine similarity table from list of vectors"""
    cos_sims = cosine_similarity(list_of_features)
    print(cos_sims)
    cos_table = pd.DataFrame(cos_sims, columns=list_of_ids, index=list_of_ids)
    # turn the cos sim into dataframe
    return cos_table


def get_k_most_similar_to(df, given_img, k_similar=5):
    # this gets the rows
    k_images = df[given_img].sort_values(ascending=False)[1:k_similar+1].index
    # this get the corresod=nding scores
    k_images_score = df[given_img].sort_values(ascending=False)[1:k_similar+1]
    for img in range(len(k_images)):
        original = k_images[i]
        plt.imshow(original)
        plt.show()
        print("Similarity score:", k_images_score[i])

    def run_all(filepath):
    print("======================STARTING============================")
    all_images = get_all_image_ids(filepath)
    print("some examples: ", all_images[:20])
    # we preprocess the image to make it ready to pass to vgg model
    print("prepocess")
    all_processed_images = []
#     all_images = np.random.choice(all_images, 5)
#     all_images = all_images[:5]
#     gen = generate_vectors(all_images)
    for image in all_images:
        processed_image = preprocess_image_for_vgg(filepath, image)
        all_processed_images.append(processed_image)

    print("Now we have preprocessed image for vgg")
    all_feature_vectors = []
    for image in all_processed_images:
        # create feature vector
        feature_vector = get_feature_vector_from_vgg(image)
#         print("making vgg")
        # reshape it before we put into
#         print("feature shape:", feature_vector.shape)
        feature_vector = reshape_feature_vector(feature_vector)
#         print("after reshaping", feature_vector)

        # now append it to all features list
        all_feature_vectors.append(feature_vector.flatten())

    print("Now we have all feature vector!")
    all_feature_vectors = np.array(all_feature_vectors)
    print("shape of the feature vector")
    print(all_feature_vectors.shape)
#     create cos sim table

    print("we have list of features ready to pass cosine similarity!")
    print("passing...")
    cos_df = create_cosine_table(all_images, all_feature_vectors)
    if len(cos_df) < 2000:
        cos_df.to_csv('cos_similarity_small_table.csv',
                      sep='\t', encoding='utf-8')
    else:
        cos_df.to_csv('cos_similarity_table.csv', sep='\t', encoding='utf-8')
    return cos_df
#     return all_feature_vectors
if __name__ == "__main__":
    image_rec = ImageRecommender(filepath)
    filepath = "../fashion-product-images-small/images/"
    ll_image_ids = get_all_image_ids(filepath)
