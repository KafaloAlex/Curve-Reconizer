import os
import sys
import json
import datetime
import numpy as np
import skimage.draw # Install skimage
import cv2 # Install opencv
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils


BASE_DIR = "C:\\Users\\hp\\Desktop\\Data Science\\Projet Alain\\Curve_Reconizer"

sys.path.append(BASE_DIR)
COCO_WEIGHTS_PATH = os.path.join(BASE_DIR, "mask_rcnn_coco.h5") # lien_de_telechargement : https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
DEFAULT_LOGS_DIR = os.path.join(BASE_DIR, "logs")


class CustomConfig(Config):
    '''Configuration pour entrainer notre dataset customisé'''

    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Nombre de classes = Background + success et failure
    STEPS_PER_EPOCH = 10 # Nombre d'étapes d'apprentissage par epoch
    DETECTION_MIN_CONFIDENCE = 0.9 # Sauter les détections avec un niveau de confiance < 90


class CustomDataset(utils.Dataset):
    
    def load_custom(self, dataset_dir, subset):

        self.add_class("object", 1, "success")
        self.add_class("object", 2, "failure")

        # Train Dataset or Validation Dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations1 = json.load(open('C:\\Users\\hp\\Desktop\\Data Science\\Projet Alain\\Curve_Reconizer\\datasets\\train\\train_annoted.json'))
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # Obtention des coordonnées x, y des points des polygones qui composent le contour de chaque instance d'objet.
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"success": 1,"failure": 2}

            num_ids = [name_dict[a] for a in objects]
            print("numids",num_ids)

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )
            
    def load_mask(self, image_id):
        """Générer des masques d'instance pour une image.
       Returns:
        masks:  Un tableau bool de forme [hauteur, largeur, nombre d'instances] avec un masque par instance.
        class_ids: un tableau 1D d'ID de classe des masques d'instance.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convertir les polygones en un masque bitmap de forme
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Retourne le chemin de l'image"""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    '''Entrainement du modèle'''
    train_path = 'C:\\Users\\hp\Desktop\\Data Science\\Projet Alain\\Curve_Reconizer\\datasets\\train'
    validation_path = 'C:\\Users\\hp\Desktop\\Data Science\\Projet Alain\\Curve_Reconizer\\datasets\\val'
    
    # Training Set
    train_dataset = CustomDataset()
    train_dataset.load_custom(train_path, 'train')
    train_dataset.prepare()

    # Validation Set
    validation_set = CustomDataset()
    validation_set.load_custom(validation_path, 'val')

    '''
        Ceci est un entraînement de base, vous pourrez le personnaliser en fonction de vos besoins.
        Vu que nous avons un petit dataset, il n'est pas nécessaire d'entrainer toutes les couches, 
        nous allons donc entraîner uniquement les entêtes
    '''

    print("Entraînement du réseau")
    model.train(train_dataset, validation_set, learning_rate=Config.LEARNING_RATE, epochs=20, layers='heads')

config = CustomConfig()
model = modellib.MaskRCNN(mode='training', config=config, model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH

if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'
])

train(model)