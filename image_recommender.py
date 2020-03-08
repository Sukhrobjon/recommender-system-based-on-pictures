

from keras.preprocessing.image import load_img
# from PIL import Image, load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import csv
import os
# load the model
# model = VGG16(weights='imagenet', include_top=False)


class ImageRecommender():
    """
    Class makes a recommendation for image
    """
    def __init__(self):
        """
        Initialize the image recommender
        """
        # self.model = model
        # self.filepath = filepath
    
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

    def get_all_image_ids(self, filepath):
        images_ids = []
        # listing all the image names into list they are ids of the image
        for img in os.listdir(filepath):
            if img.endswith('jpg'):
                images_ids.append(img)

        return images_ids

    def read_img_ids_from_file(self, filename):
        """
        reads the the image ids from preselected 2000 image ids
        """
        img_ids = []
        with open(filename) as file:
            for line in file:
                line = line.strip().split("\n")
                line = "".join(str(char) for char in line)
                img_ids.append(line)

        return img_ids

    # reshape the feature_vector so we can find cosine similarity score
    def reshape_feature_vector(self, feature_vector):
        """
        Takes 4d array, return 2D representation
        """
        # this is the vgg output 7, 7, 512
        return feature_vector.reshape(1, -1)

    def create_cosine_table(self, list_of_ids, list_of_features):
        """Creates a cosine similarity table from list of vectors"""
        print(list_of_features.shape)
        cos_sims = cosine_similarity(list_of_features)
        print(cos_sims)
        cos_table_df = pd.DataFrame(cos_sims, columns=list_of_ids, index=list_of_ids)
        # turn the cos sim into dataframe
        # save it to file
        cos_table_df.to_csv('cos_sim_2000_samples.csv', sep='\t', encoding='utf-8')
        
        return cos_table_df

    def get_k_most_similar_to(self, df, given_img, k_similar=5):
        # this gets the rows
        k_images = df[given_img].sort_values(ascending=False)[
            1:k_similar+1].index
        # this get the corresod=nding scores
        k_images_score = df[given_img].sort_values(ascending=False)[
            1:k_similar+1]
        for i in range(len(k_images)):
            original = k_images[i]
            plt.imshow(original)
            plt.show()
            print("Similarity score:", k_images_score[i])

    def build_image_recommender_v1(self, filepath):
        """
        Builds image recommender
        """
        # sample size is 2000 now
        all_image_ids = self.read_img_ids_from_file('smaller_chunk_of_samples.txt')

        print(len(all_image_ids)
        
        feature_vector = []

        # all_image_ids = all_image_ids[:5]
        
        for index, img_id in enumerate(all_image_ids):
            # prepocess the image
            if index == 500 or index == 1000 or index == 1500:
                print("index: ", index)

            
            processed_img = self.preprocess_image_for_vgg(filepath, img_id)
            # reshape it
            
            # extract features
            feature = self.get_feature_vector_from_vgg(processed_img)
            feature = self.reshape_feature_vector(feature)
            # flatten it before to build feature vector
            feature = feature.flatten()
            feature_vector.append(feature)
        print("feature vector")
        print(feature_vector)
        print("finished extracting features now making cosine")
        feature_vector = np.array(feature_vector)
        # build a cos table
        cosine_sim_dataframe = self.create_cosine_table(all_image_ids, feature_vector)
        # save it 
        print("built cosine table")
        print("DONE!!!!!!")
        return cosine_sim_dataframe



   
filepath =  "../fashion-product-images-small/images/"
obj = ImageRecommender()
obj.build_image_recommender_v1(filepath)
