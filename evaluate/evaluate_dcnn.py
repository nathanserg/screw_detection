import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from PIL import Image
import time

import tensorflow as tf
from tensorflow import keras
from keras.applications import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

os.getcwd()

model = DenseNet201
model_name = 'DenseNet201'
weight_directory = 'weights' + f'/{model_name}_final.h5'


NUM_CLASSES   = 6
IMAGE_SIZE    = (71, 71)


class NNModel:    
    def loadmodel(self, path, model_name, IMAGE_SIZE, NUM_CLASSES):
        l2 = keras.regularizers.l2(0.01)
        base_model = model_name(include_top=False,
                                weights='imagenet',
                                input_tensor=None,
                                input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

        base_model.trainable = False

        # add a global spatial average pooling layer
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2)(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2)(x)
        x = Dropout(0.1)(x)

        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)

        # this is the model we will train
        model = keras.Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(path)
        return model
  
    def __init__(self, path, model_name,IMAGE_SIZE, NUM_CLASSES):
       self.model = self.loadmodel(path, model_name, IMAGE_SIZE, NUM_CLASSES)

    def predict(self, X):
        return self.model.predict(X)

class ScrewDetector:
    def __init__(self, model_name, weight_directory, IMAGE_SIZE, NUM_CLASSES):
        # to keep the singular and multiple screw data
        self.screw = dict()
        self.screw_data = []
        self.labels = ['allen', 'hexagonal', 'nas', 'philipsAndPZ', 'slotted', 'torx']
        # self.labels = ['flat', 'nas', 'philips', 'torx']
        
        # define keras models
        self.model1 = NNModel(weight_directory, model_name, IMAGE_SIZE, NUM_CLASSES)
        
    def match_center(self, elt, centers):
        """Fonction qui permet d'éliminer les contours doublons en calculant si deux contours ont un centre identique

        Parameters
        ----------
        elt : tuple (2 values)
            L'élément à traiter
        centers : list
            La liste de tous les centres des contours retenus jusqu'à présent

        Returns
        -------
        bool
            True si déjà présent \n
            False si absent
        """
        tol = 20 # Tolérance (en pixels) de distance entre deux centres
        for centre in centers :
            if abs(elt[0] - centre[0]) <= tol and abs(elt[1] - centre[1]) <= tol :
                return True
        return False 

    def extract(self, frame, centers) :
        """Fonction qui sauvegarde toutes les formes rondes/hexagonales dans une frame

        Parameters
        ----------
        frame : cv2 image
            La frame à analyser
        centers : list(tuple(cx, cy, radius))
            La liste des centres et rayons des cercles et hexagones détectés dans la frame
        """
        ImgToClassify = []
        ImgData = []
        # Passage de BGR à RGB pour l'interprétation de PIL
        #frame = frame[:, :, ::-1]
        img = Image.fromarray(frame) # Conversion au format de la librairie PIL
        compteur = 0 # Compteur pour l'enregistrement
        for elt in centers :
            compteur += 1
            timestamp = time.strftime("%H-%M-%S")
            # On crop la frame autour du centre du cercle/de l'hexagone, aux dimensions de la forme détectée (rayon) et avec une marge de 5% au-dessus
            crop_rect = (elt[0]-elt[2]-0.05*elt[2], elt[1]-elt[2]-0.05*elt[2], elt[0]+elt[2]+0.05*elt[2], elt[1]+elt[2]+0.05*elt[2])
            #print(crop_rect)
            cropped = img.crop(crop_rect)
            #cropped.save(os.path.join("tests", f'{compteur}.jpg'))
            np_cropped = np.array(cropped)
            #print(type(np_cropped))
            # On sauvegarde l'image cropée
            np_cropped = cv2.resize(np_cropped, IMAGE_SIZE)
            np_image_data = np_cropped/255
            np_image_data = np.stack((np_image_data,)*3, axis=-1)
            ImgToClassify.append(np_image_data)
            ImgData.append([np_image_data, elt[0], elt[1]])
        ImgToClassify = np.array(ImgToClassify)
        return ImgToClassify, ImgData   
    
    def OpencvDectection(self, img):
        # Création d'une copie, afin de traiter une frame et en conserver une propre pour l'affichage
        frame = img.copy()
        # Passage en Niveaux de gris
        gray = img.copy()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        L=[]
        H = []

        centers = []

        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 80, param1 = 100, param2 = 70, minRadius = 10, maxRadius = 75)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                centers.append((x, y, r, 0))

        # 3 niveaux de seuils de traitements pour augmenter la fiabilité de la détection
        thresholds = [[300,400],[100, 200],[200, 300],[50, 100]]

        contours = []
        contours_dil = []
        for i in range(len(thresholds)) :
            # Exécution de l'algorithme Canny qui passe d'une image à simplement des vecteurs "outline" des objets avec leqs seuls actuels de la boucle
            thresh = cv2.Canny(gray, thresholds[i][0], thresholds[i][1])
            if i == 0 :
                tracage = np.copy(thresh)
            else :
                tracage += thresh

        kernel = np.ones((3, 3), np.uint8)
        tracage_dil = cv2.dilate(tracage, kernel, iterations=1)
        # Détection des contours sur l'image seuillée
        cont, res = cv2.findContours(tracage, 1, 2)
        cont_dil, res = cv2.findContours(tracage_dil, 1, 2)
        for elt in cont :
            # Calcul de l'aire du contour (car parfois la détection de contours renvoie des lignes)
            area = cv2.contourArea(elt)
            if area > 200 :
                # On ne garde que si le contour a une surface réaliste
                contours.append(elt)
        for elt in cont_dil :
            # Calcul de l'aire du contour (car parfois la détection de contours renvoie des lignes)
            area = cv2.contourArea(elt)
            if area > 200 :
                # On ne garde que si le contour a une surface réaliste
                contours_dil.append(elt)
        
        
        for CONTOUR in (contours, contours_dil) :
            #print(len(CONTOUR))
            for cnt in CONTOUR:
                # Stockage des crd d'origine du contour en cours de traitement
                x1,y1 = cnt[0][0]
                # Approximation des contours détectés à des polygones avec une simplification à 1% de la taille (si valeur plus élevée, détection des cercles compromise)
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
                if len(approx)>10: # Un polygone de plus de 10 côtés peut être assimilé à un cercle, mais on fait davantage de vérifications ensuite
                    continue # On ne veut pas vérifier s'il s'agit d'un hexagone si le contours a déjà été classé comme cercle


                # Approximation des contours détectés à des polygones avec une simplification à 4% de la taille (permet une bonne détection des hexagones, valeur trouvée par tatonnement)
                approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
                if (len(approx) >=6 and len(approx) <= 10) : # Un polygone de 6 côtés est un hexagone, mais on fait davantage de vérifications ensuite
                    # On vérifie que le contours est convexe
                    k = cv2.isContourConvex(approx)
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    if self.match_center((cx, cy), centers) :
                        continue
                    if k and radius > 10 :
                        H.append([2,np.array([[cx],[cy]]), radius])
                        cv2.putText(img, 'Hexagon '+str(len(H)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        thresh = cv2.drawContours(thresh, [cnt], -1, (255,255,0), 5)
                        img = cv2.drawContours(img, [cnt], -1, (255,255,0), 5)
                        centers.append((cx, cy, radius, 1))
        return frame, centers

    def draw_square_and_class(self, image, centers_element, class_element):
        # Unpack the element
        cx, cy, radius, _ = centers_element

        # Calculate square coordinates
        x1 = int(cx - radius)
        y1 = int(cy - radius)
        x2 = int(cx + radius)
        y2 = int(cy + radius)

        if class_element == 'nas':
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        else :
            # Draw a square on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # (0, 255, 0) is the color (here, green), and 2 is the thickness
            # Put the text in the top-left corner of the rectangle
            cv2.putText(image, class_element, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        return image

    def collect_or_detect_screw(self,img, ori_img, file):
        # get a copy and resize it
        result = ori_img.copy()

        frame, centers = self.OpencvDectection(img)

        self.img_to_classify, self.img_data = self.extract(frame, centers)

        img_raw = self.img_to_classify.copy()

        self.predicted1 = self.model1.predict(img_raw)

        # for i in predicted1:
        #     print(self.labels[np.argmax(i)])

        #print(self.predicted1)
        for i, v in enumerate(centers):
            self.draw_square_and_class(result, v, self.labels[np.argmax(self.predicted1[i])])

        result = cv2.resize(result, (1200,900))
        # cv2.imshow('result', result)
        # cv2.waitKey(0)  # 0 means wait indefinitely for a key press
        # cv2.destroyAllWindows()
        # result.save(f'evaluate/DCNN_predictions/{file}.jpg')
        cv2.imwrite(f'evaluate/DCNN_predictions_2/{file}.jpg', result)
        



if __name__ == "__main__":
    print("Running ...")
    folder_path = os.path.abspath('evaluate/Photos')
    files = [os.path.splitext(filename)[0] for filename in os.listdir(folder_path)]

    print("Classes : allen, hexagonal, nas, philipsAndPZ, slotted, torx")
    # print("Classes : flat, nas, philips, torx")

    for file in files:
        print(file)
        ori_img = cv2.imread(f'evaluate/Photos/{file}.jpg')
        img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        screwdetector = ScrewDetector(model, weight_directory, IMAGE_SIZE, NUM_CLASSES)
        screwdetector.collect_or_detect_screw(img, ori_img, file)
    print("Done")
    