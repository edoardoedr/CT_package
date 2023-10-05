import numpy as np
from .CT_stack_class import CT_stack
from .utils import *
import copy
import pprint
import json
from PIL import Image, ImageFilter, ImageOps, ExifTags
import statistics as st

keys_classe = ["directory_principale", "nome", "images", "labels", "directory_pesi", "numero_classi", "numero_GPU", "network_name", "lista_filtri_grays", "lista_filtri_labels", "modalità", "prediction", "intersection_over_union", "explained"]

############################ INIZIO CLASSE ##############################################################
class CT_dataset:

    def __init__(self, data_dir = None, step = None, images = [], labels = []):

        self.dataset_info = dict.fromkeys(keys_classe)
        self.dataset_info["images"] = []
        self.dataset_info["labels"] = []
        self.dataset_info["lista_filtri_grays"] = []
        self.dataset_info["lista_filtri_labels"] = []
        self.dataset_info["prediction"] = []
        self.dataset_info["intersection_over_union"] = []
        self.dataset_info["explained"] = "No"

        if data_dir is not None:
            self.images, self.labels = self.upload_dataset(data_dir, step)
            self.dataset_info["directory_principale"] = os.path.dirname(data_dir.rstrip("/")) + "/"
            nome = data_dir.rstrip("/")
            nome = nome.rsplit("/", 1)[-1]
            self.dataset_info["nome"] = nome
            for name_images in self.images:
                    self.dataset_info["images"].append(name_images.name)
            
        # Altrimenti, usa lo stack fornito come parametro
        else:
            self.images = images
            self.labels = labels

        self.prediction = []
        self.grad_CAM = []

        # controlla se lo stack images è una classe di tipo CT_dataset
        assert all(isinstance(x, CT_stack) for x in images), "Images must be a images stack"
        
        if self.prediction != []:
              assert all(isinstance(x, CT_stack) for x in self.prediction), "predictions must be a images stack or a empty list"
        
        if self.grad_CAM != []:
              assert all(isinstance(x, CT_stack) for x in self.grad_CAM), "grad_CAM must be a images stack or a empty list"
              
        if self.labels != []:
              assert all(isinstance(x, CT_stack) for x in self.labels), "Labels must be a images stack or a empty list"
              for name_labels in self.labels:
                    self.dataset_info["labels"].append(name_labels.name)

        if "train" in self.dataset_info["nome"] or "Train" in self.dataset_info["nome"]:
                assert all(isinstance(x, CT_stack) for x in self.labels), "Labels must be a images stack in train dataset"
                assert len(self.images) == len(self.labels), "Labels must have the same len of images in train dataset"
                self.dataset_info["modalità"] = "train"
        elif "test" in self.dataset_info["nome"] or "Test" in self.dataset_info["nome"]:
                if self.labels == []:
                    self.dataset_info["modalità"] = "test_predici"
                else:
                    assert all(isinstance(x, CT_stack) for x in self.labels), "Labels must be a images stack in test_performance dataset"
                    assert len(self.images) == len(self.labels), "Labels must have the same len of images in test_performance dataset"
                    self.dataset_info["modalità"] = "test_performance"

    def upload_dataset(data_dir, step):
        files_or_folders = os.listdir(data_dir)
        files_or_folders.sort()
        if len(files_or_folders) == 2 and (splitted_train(data_dir) == True or splitted_test(data_dir) == True):
            lista_stack = os.listdir(data_dir + files_or_folders[0] + "/")
            lista_stack.sort()
            lista_labels = os.listdir(data_dir + files_or_folders[1] + "/")
            lista_labels.sort()
            path_images = []
            path_labels = []
            if len(lista_labels) == len(lista_stack):
                for c, d in zip(lista_stack, lista_labels):
                    temp_c = c.rsplit(".", -1)[0]
                    temp_d = d.rsplit(".", -1)[0]
                    if temp_c not in ("label_" + temp_d):
                        raise TypeError("Non c'è corrispondenza tra images e labels")
                    else:
                        path_images.append(data_dir + files_or_folders[0] + "/" + c)
                        path_labels.append(data_dir + files_or_folders[1] + "/" + d)
                immagini = []
                etichette = []
                for file_images, file_labels in zip(path_images, path_labels):
                    immagini.append(CT_stack.read_stack(file_images, step))
                    etichette.append(CT_stack.read_stack(file_labels, step))

                return immagini, etichette
            else:
                for c in lista_stack:
                    path_images.append(data_dir + files_or_folders[0] + "/" + c)
                immagini = []
                etichette = []
                for file_images in path_images:
                    immagini.append(CT_stack.read_stack(file_images, step))

                return immagini, etichette

        else:
            path_images = []
            path_labels = []
            for files in files_or_folders:
                if "label" not in files:
                    path_images.append(data_dir + files)
                else:
                    path_labels.append(data_dir + files)

            immagini = []
            etichette = []
            if len(path_images) == len(path_labels):
                for files_grays, file_labels in zip(path_images, path_labels):
                    nome_grays = files_grays.split("/")[-1]
                    nome_grays = nome_grays.rsplit(".", -1)[0]
                    nome_labels = file_labels.split("/")[-1]
                    nome_labels = nome_labels.rsplit(".", -1)[0]
                    if nome_grays not in nome_labels:
                        raise TypeError("Non c'è corrispondenza tra images e labels")
                    else:
                        immagini.append(CT_stack.read_stack(files_grays, step))
                        etichette.append(CT_stack.read_stack(file_labels, step))

                return immagini, etichette
            else:
                for files_grays in path_images:
                    nome_grays = files_grays.split("/")[-1]
                    nome_grays = nome_grays.rsplit(".", -1)[0]
                    immagini.append(CT_stack.read_stack(files_grays, step))

                return immagini, etichette
    
    @staticmethod
    def merge(dataset_uno, *args):
        lista_images = dataset_uno.images
        lista_labels = dataset_uno.labels
        lista_prediction = dataset_uno.prediction
        lista_grad_CAM = dataset_uno.grad_CAM

        images_list = dataset_uno.dataset_info["images"]
        labels_list = dataset_uno.dataset_info["labels"]
        lista_filtri_grays = dataset_uno.dataset_info["lista_filtri_grays"]
        lista_filtri_labels = dataset_uno.dataset_info["lista_filtri_labels"]
        prediction_list = dataset_uno.dataset_info["prediction"]

        for datas in args:
            for img in datas.images:
                lista_images.append(img)
            for lbl in datas.labels:
                lista_labels.append(lbl)
            for pred in datas.prediction:
                lista_prediction.append(pred)
            for grad in datas.grad_CAM:
                lista_grad_CAM.append(grad)
            images_list.append(datas.dataset_info["images"])
            labels_list.append(datas.dataset_info["labels"])
            lista_filtri_grays.append(datas.dataset_info["lista_filtri_grays"])
            lista_filtri_labels.append(datas.dataset_info["lista_filtri_labels"])
            prediction_list.append(datas.dataset_info["prediction"])
                        
        
        #print(len(lista_images))
        #print(len(lista_labels))
        new_dataset = CT_dataset(images = lista_images, labels = lista_labels)
        new_dataset.prediction = lista_prediction
        new_dataset.grad_CAM = lista_grad_CAM
        new_dataset.dataset_info = copy.deepcopy(dataset_uno.dataset_info)
        new_dataset.dataset_info["images"] = images_list
        new_dataset.dataset_info["labels"] = labels_list
        new_dataset.dataset_info["lista_filtri_grays"] = lista_filtri_grays
        new_dataset.dataset_info["lista_filtri_labels"] = lista_filtri_grays
        new_dataset.dataset_info["prediction"] = prediction_list

        return new_dataset

    def copy_dataset(self):

        new_lista_images = []
        new_lista_labels = []
        new_lista_prediction = []
        new_lista_grad_CAM = []

        for img in self.images:
                new_lista_images.append(img.copy_stack())
        for lbl in self.labels:
                new_lista_labels.append(lbl.copy_stack())
        for pred in self.prediction:
                new_lista_prediction.append(pred.copy_stack())
        for grad in self.grad_CAM:
                new_lista_grad_CAM.append(grad.copy_stack())

        new_dataset = CT_dataset(images = new_lista_images, labels = new_lista_labels)
        new_dataset.dataset_info = copy.deepcopy(self.dataset_info)
        new_dataset.prediction = new_lista_prediction
        new_dataset.grad_CAM = new_lista_grad_CAM

        return new_dataset

    def change_values(self, valori, substitute, tipologia ="label"):
            if tipologia == "label":
                dati_cambiati = []
                for dati in self.labels:
                    dati_cambiati.append(dati.change_values(valori, substitute))
                self.labels = dati_cambiati

            elif tipologia == "image":
                dati_cambiati = []
                for dati in self.images:
                    dati_cambiati.append(dati.change_values(valori, substitute))
                self.images = dati_cambiati
            
            elif tipologia == "prediction":
                dati_cambiati = []
                for dati in self.prediction:
                    dati_cambiati.append(dati.change_values(valori, substitute))
                self.prediction = dati_cambiati

    def label_x_images(self):
        new_images = []
        for img, lbl in zip(self.images, self.labels):
            new_img = CT_stack.multiply(img,lbl)
            new_images.append(new_img)

        self.images = new_images

    def save_dataset(self):
    
        dest_dir = self.dataset_info["directory_principale"] + self.dataset_info["nome"] + "/"
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        if not os.path.exists(dest_dir + "images/"):
            os.mkdir(dest_dir + "images")
        if not os.path.exists(dest_dir + "labels/"):
            os.mkdir(dest_dir + "labels")

        for img, labl in zip(self.images, self.labels):
            img.save_stack(dest_dir = dest_dir + "images/", estenzione = ".nii.gz")
            labl.save_stack(dest_dir = dest_dir + "labels/", estenzione = ".nii.gz")

        print("Dataset saved")
    
    def crop_ROI_dataset(self):

        new_images = []
        new_labels = []
        for image, label in zip(self.images, self.labels):
            coordinate = where_crop_ROI(label.np_stack)
            new_images.append(image.crop_ROI(coordinate=coordinate))
            new_labels.append(label.crop_ROI(coordinate=coordinate))

        self.images = new_images
        self.labels = new_labels

    def resize_dataset(self, new_size = None, scale = None):
        new_images = []
        new_labels = []
        for image, label in zip(self.images, self.labels):
            new_images.append(image.resize_stack(new_size, scale))
            new_labels.append(label.resize_stack(new_size, scale))

        self.images = new_images
        self.labels = new_labels

    def mode_filter_25D(self, filter_size, tipologia="label"):
        if tipologia == "label":
            dati_filtrati = []
            for dati in self.labels:
                dati_filtrati.append(dati.mode_filter_25D(filter_size))
            self.labels = dati_filtrati
            self.dataset_info["lista_filtri_labels"].append("mode")

        elif tipologia == "image":
            dati_filtrati = []
            for dati in self.images:
                dati_filtrati.append(dati.mode_filter_25D(filter_size))
            self.images = dati_filtrati
            self.dataset_info["lista_filtri_grays"].append("mode")
        
        elif tipologia == "prediction":
            dati_filtrati = []
            for dati in self.prediction:
                dati_filtrati.append(dati.mode_filter_25D(filter_size))
            self.prediction = dati_filtrati

    def median_filter_25D(self, filter_size, tipologia="label"):
        if tipologia == "label":
            dati_filtrati = []
            for dati in self.labels:
                dati_filtrati.append(dati.median_filter_25D(filter_size))
            self.labels = dati_filtrati
            self.dataset_info["lista_filtri_labels"].append("median")

        elif tipologia == "image":
            dati_filtrati = []
            for dati in self.images:
                dati_filtrati.append(dati.median_filter_25D(filter_size))
            self.images = dati_filtrati
            self.dataset_info["lista_filtri_grays"].append("median")
        
        elif tipologia == "prediction":
            dati_filtrati = []
            for dati in self.prediction:
                dati_filtrati.append(dati.median_filter_25D(filter_size))
            self.prediction = dati_filtrati

    def unsharpmask(self):
        dati_filtrati = []
        for dati in self.images:
            dati_filtrati.append(dati.unsharpmask())
        self.images = dati_filtrati
        self.dataset_info["lista_filtri_grays"].append("unsharp")

    def sharpen(self):
        dati_filtrati = []
        for dati in self.images:
            dati_filtrati.append(dati.sharpen())
        self.images = dati_filtrati
        self.dataset_info["lista_filtri_grays"].append("sharpen")

    def fill_holes(self, label_riempire, tipologia = "prediction"):
        if tipologia == "label":
            dati_riempiti = []
            for dati in self.labels:
                dati_riempiti.append(dati.fill_holes(label_riempire))
            self.labels = dati_riempiti
            self.dataset_info["lista_filtri_labels"].append("fill_holes")
        elif tipologia == "prediction":
            dati_riempiti = []
            for dati in self.prediction:
                dati_riempiti.append(dati.fill_holes(label_riempire))
            self.prediction = dati_riempiti
    
    def delete_stack(self, name):
        cont = 0
        lista_eliminare = []
        for image in self.images:
            if image.name == name:
                lista_eliminare.append(cont)
            cont = cont + 1
        for elmn in lista_eliminare:
            self.images.pop(elmn)
            self.labels.pop(elmn)
    
    def inizializza_test(self, cls_or_dir):

        old_dict = copy.deepcopy(self.dataset_info)

        if isinstance(cls_or_dir, CT_dataset):
            self.dataset_info = copy.deepcopy(cls_or_dir.dataset_info)
    
        elif isinstance(cls_or_dir, str):
            file = open(cls_or_dir, "r")
            self.dataset_info = json.load(file)

        if self.labels == []:
                    self.dataset_info["modalità"] = "test_predici"
        else:
                assert all(isinstance(x, CT_stack) for x in self.labels), "Labels must be a images stack in test_performance dataset"
                assert len(self.images) == len(self.labels), "Labels must have the same len of images in test_performance dataset"
                self.dataset_info["modalità"] = "test_performance"
        
        self.dataset_info["images"] = old_dict["images"]
        self.dataset_info["labels"] = old_dict["labels"]

    def performance(self):
        if self.dataset_info["modalità"] == "test_performance":
            ious = []
            for lab, pred in zip(self.labels, self.prediction):
                iou = IOU3D(pred.stack_class, lab.stack_class, list(range(1, self.dataset_info["numero_classi"])))
                print(iou)
                ious.append(iou)
            self.dataset_info["intersection_over_union"].append(ious)

    def save_prediction(self):
        dest_dir = os.path.dirname(self.dataset_info["directory_pesi"]) + "/"
        for preds in self.prediction:
            preds.save_stack(dest_dir, ".nii.gz")
            
    def save_grad_CAM(self):
        dest_dir = os.path.dirname(self.dataset_info["directory_pesi"]) + "/"
        for preds in self.grad_CAM:
            preds.save_stack(dest_dir, ".nii.gz")

    def print_info(self):
        pprint.pprint(self.dataset_info)

    def save_info(self):
        with open(self.dataset_info["directory_principale"] + self.dataset_info["nome"] + ".json", "w") as file:
            json.dump(self.dataset_info, file)

    def split_dataset(self):

        dest_dir = self.dataset_info["directory_principale"] + "dataset_training/"
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        if not os.path.exists(dest_dir + "train/"):
            os.mkdir(dest_dir + "train")
        if not os.path.exists(dest_dir + "val/"):
            os.mkdir(dest_dir + "val")
        if not os.path.exists(dest_dir + "train/img/"):
            os.mkdir(dest_dir + "train/img")
        if not os.path.exists(dest_dir + "train/label/"):
            os.mkdir(dest_dir + "train/label")
        if not os.path.exists(dest_dir + "val/img/"):
            os.mkdir(dest_dir + "val/img")
        if not os.path.exists(dest_dir + "val/label/"):
            os.mkdir(dest_dir + "val/label")

        file_rimuovere = os.listdir(dest_dir + "train/img/")
        for c in range(len(file_rimuovere)):
            os.remove(dest_dir + "train/img/" + file_rimuovere[c])
        file_rimuovere = os.listdir(dest_dir + "train/label/")
        for c in range(len(file_rimuovere)):
            os.remove(dest_dir + "train/label/" + file_rimuovere[c])
        file_rimuovere = os.listdir(dest_dir + "val/img/")
        for c in range(len(file_rimuovere)):
            os.remove(dest_dir + "val/img/" + file_rimuovere[c])
        file_rimuovere = os.listdir(dest_dir + "val/label/")
        for c in range(len(file_rimuovere)):
            os.remove(dest_dir + "val/label/" + file_rimuovere[c])

        for images, labels in zip(self.images, self.labels):
            split_train(images.np_stack, labels.np_stack, dest_dir, images.name)
