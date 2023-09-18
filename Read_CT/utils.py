import numpy as np
import cv2
import os
from PIL import Image, ImageFilter, ImageOps, ExifTags
import statistics as st
from scipy import ndimage, signal
import random
from multiprocessing import Pool, cpu_count
import time
from functools import partial


def filtro_mode(filter_size, immagine_np):
    immagine = Image.fromarray(immagine_np)
    filtrata = immagine.filter(ImageFilter.ModeFilter(size=filter_size))
    filtrata_np = np.copy(filtrata)

    return filtrata_np

def FiltroUnsharp(immagine):
    immagine = Image.fromarray(immagine)
    immagine_filtrata = immagine.filter(ImageFilter.UnsharpMask(radius=6, percent=350, threshold=1))
    immagine = np.copy(immagine_filtrata)

    return immagine

def fill_holed2D(value_to_fill, structure, immagine_np):
    for value in value_to_fill:
        imm_value = np.copy(immagine_np)
        imm_original = np.copy(immagine_np)
        imm_original[imm_original == value] = 0
        imm_value[imm_value != value] = 0
        imm_value[imm_value == value] = 1
        imm_value = ndimage.binary_fill_holes(imm_value, structure=structure).astype(np.uint8)
        imm_value[imm_value == 1] = value
        imm_original[imm_value == value] = value

    return imm_original

def Filtrosharpen(immagine):
    kernel = [-1, -1, -1, -1, 9, -1, -1, -1, -1]
    kernel = np.reshape(kernel, (3, 3))
    sharpened_image = ndimage.convolve(immagine, kernel)
    immagine = np.copy(sharpened_image)

    return immagine

def IOU3D(stack_prediction, stack_label, classi):
    t = 0
    iou_classi = np.zeros(len(classi) + 1)
    percentuale_classi = np.zeros(len(classi))

    for value in classi:
        print(f"Calcolo IOU per la classe {value}", end="\r")
        print('\n', end="\r")

        prediction_classe = np.copy(stack_prediction)
        prediction_classe[prediction_classe != value] = 0

        label_classe = np.copy(stack_label)
        label_classe[label_classe != value] = 0

        union_classe = prediction_classe + label_classe
        intersection_classe = prediction_classe * label_classe

        union_classe[union_classe > value] = value
        intersection_classe[intersection_classe > value] = value

        sum_union_classe = (union_classe == value).sum()
        sum_intersection_classe = (intersection_classe == value).sum()

        iou_classi[t] = sum_intersection_classe / sum_union_classe
        percentuale_classi[t] = (stack_label == value).sum()
        t = t + 1

    if len(percentuale_classi) > 1:
        totale_percentuale = np.sum(percentuale_classi)
        for n in range(len(percentuale_classi)):
            percentuale_classi[n] = percentuale_classi[n] / totale_percentuale

        iou_classi[t] = 0
        for i in range(len(percentuale_classi)):
            iou_classi[t] = iou_classi[t] + iou_classi[i] * percentuale_classi[i]

        return iou_classi
    else:
        return iou_classi[0]

    return None


def calcola_moda(array):
    zeta = array.shape[2]
    rig = array.shape[0]
    col = array.shape[1]
    moda = np.empty(zeta, dtype=np.uint8)
    new_array = np.empty([rig, col], dtype=np.uint8)

    for r in range(rig):
        for c in range(col):
            moda = np.copy(array[r, c, :])
            new_array[r, c] = np.copy(st.mode(moda))

    return new_array

def calcola_moda_parallelized(stacks):
    start_time = time.perf_counter()
    z = stacks[0].shape[2]
    rig = stacks[0].shape[0]
    col = stacks[0].shape[1]
    num_stack = len(stacks)
    pila = np.empty([rig, col, num_stack], dtype=np.uint8)
    new_stack = np.empty([rig, col, z], dtype=np.uint8)

    lista_a3 = []

    for x in range(0, z, 1):
        immagini = []
        for i in range(len(stacks)):
            stac = stacks[i]
            immagini.append(stac[:, :, x])
        pila = np.dstack((immagine for immagine in immagini))
        # print(pila.shape)
        lista_a3.append(pila)

    with Pool() as pool:
        result = pool.map(calcola_moda, lista_a3)

    for n in range(z):
        new_stack[:, :, n] = np.copy(result[n])

    finish_time = time.perf_counter()
    print("Moda calcolata in {} seconds - using multiprocessing".format(finish_time - start_time))
    print("---")

    return new_stack

def where_crop_ROI(array):
    z = array.shape[2]
    rig = array.shape[0]
    col = array.shape[1]

    lista_coordinate_X_min = []
    lista_coordinate_Y_min = []
    lista_coordinate_Z_min = []
    lista_coordinate_X_max = []
    lista_coordinate_Y_max = []
    lista_coordinate_Z_max = []

    for c in range(0, z, 1):
        fetta = np.copy(array[:, :, c])
        if fetta.mean() != 0:
            coords = np.argwhere(fetta != 0)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            lista_coordinate_X_min.append(x_min)
            lista_coordinate_Y_min.append(y_min)
            lista_coordinate_X_max.append(x_max)
            lista_coordinate_Y_max.append(y_max)

    for c in range(0, rig, 1):
        fetta = np.copy(array[c, :, :])
        if fetta.mean() != 0:
            coords = np.argwhere(fetta != 0)
            y_min, z_min = coords.min(axis=0)
            y_max, z_max = coords.max(axis=0)
            lista_coordinate_Z_min.append(z_min)
            lista_coordinate_Z_max.append(z_max)

    x_min = min(lista_coordinate_X_min)
    y_min = min(lista_coordinate_Y_min)
    z_min = min(lista_coordinate_Z_min)
    x_max = max(lista_coordinate_X_max)
    y_max = max(lista_coordinate_Y_max)
    z_max = max(lista_coordinate_Z_max)

    coordinate_stack = [x_min, y_min, z_min, x_max, y_max, z_max]

    return coordinate_stack


def crop_stack(stack, coordinate):
    x_min = coordinate[0]
    y_min = coordinate[1]
    z_min = coordinate[2]
    x_max = coordinate[3]
    y_max = coordinate[4]
    z_max = coordinate[5]
    new_stack = np.copy(stack[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1])

    return new_stack


def return_tointerpolate(grayscale, label):
    z = grayscale.shape[2]
    righe = grayscale.shape[0]
    colonne = grayscale.shape[1]
    indici = []
    for n in range(0, z, 1):
        conteggio_label = label[:, :, n].sum()
        if (conteggio_label == 0):
            indici.append(n)

    stack_grayscale_prevedere = np.empty([righe, colonne, len(indici)], dtype=grayscale.dtype)
    stack_label_prevedere = np.empty([righe, colonne, len(indici)], dtype=label.dtype)
    stack_grayscale_training = np.empty([righe, colonne, z - len(indici)], dtype=grayscale.dtype)
    stack_label_training = np.empty([righe, colonne, z - len(indici)], dtype=label.dtype)

    i = 0
    c = 0
    for n in range(0, z, 1):
        if (i < len(indici) and n == indici[i]):
            stack_grayscale_prevedere[:, :, i] = np.copy(grayscale[:, :, n])
            stack_label_prevedere[:, :, i] = np.copy(label[:, :, n])
            i = i + 1
        else:
            stack_grayscale_training[:, :, c] = np.copy(grayscale[:, :, n])
            stack_label_training[:, :, c] = np.copy(label[:, :, n])
            c = c + 1

    return stack_grayscale_training, stack_label_training, stack_grayscale_prevedere, stack_label_prevedere

def resample(stack, step):

    stack = stack[:,:,0:stack.shape[2]-1]
    new_stack = np.empty([stack.shape[0], stack.shape[1], int(stack.shape[2] / step)], dtype=stack.dtype)
    n = 0
    for i in range(0, stack.shape[2] - step, step):
        new_stack[:, :, n] = np.copy(stack[:, :, i])
        n = n + 1

    return new_stack


def splitted_train(data_dir):
    folders = os.listdir(data_dir)
    folders.sort()
    if os.path.isfile(data_dir + folders[0]) or os.path.isfile(data_dir + folders[1]):
        return False
    elif (folders[0] == "images" or folders[0] == "images_train") and (
            folders[1] == "labels" or folders[1] == "labels_train"):
        return True


def splitted_test(data_dir):
    folders = os.listdir(data_dir)
    folders.sort()
    if os.path.isfile(data_dir + folders[0]) or os.path.isfile(data_dir + folders[1]):
        return False
    elif (folders[0] == "images" or folders[0] == "images_test") and (
            folders[1] == "labels" or folders[1] == "labels_test"):
        return True


def split_train(grayscale, label, dest_dir, name, estenzione = ".tif"):
    lung_tot = 0
    i = 0
    while lung_tot != grayscale.shape[2]:
        len_val = int(grayscale.shape[2] * 25 / 100) - i
        len_train = int(grayscale.shape[2] - len_val)
        lung_tot = len_val + len_train
        i = i + 1

    np.random.seed(0)
    ran = np.zeros(grayscale.shape[2], dtype=int)

    grayscale_val = np.zeros([grayscale.shape[0], grayscale.shape[1], len_val], dtype=grayscale.dtype)
    grayscale_train = np.zeros([grayscale.shape[0], grayscale.shape[1], len_train], dtype=grayscale.dtype)
    label_val = np.zeros([grayscale.shape[0], grayscale.shape[1], len_val], dtype=np.uint8)
    label_train = np.zeros([grayscale.shape[0], grayscale.shape[1], len_train], dtype=np.uint8)

    x = random.randint(1, 10)
    i = 0
    while i < len_val:
        x = random.randint(0, grayscale.shape[2] - 1)
        if (ran[x] == 0):
            ran[x] = x
            i = i + 1

    i = 0
    x = 0
    for c in range(grayscale.shape[2]):
        if (i < len_train and ran[c] == 0):
            grayscale_train[:, :, i] = np.copy(grayscale[:, :, c])
            label_train[:, :, i] = np.copy(label[:, :, c])
            i = i + 1
        elif (x < len_val and ran[c] != 0):
            grayscale_val[:, :, x] = np.copy(grayscale[:, :, c])
            label_val[:, :, x] = np.copy(label[:, :, c])
            x = x + 1

    for n in range(0, grayscale_train.shape[2], 1):
        nome = name + "{:04n}".format(n)
        cv2.imwrite(dest_dir + "train/img/" + nome + estenzione, grayscale_train[:, :, n])
        cv2.imwrite(dest_dir + "train/label/" + "label_" + nome + estenzione, label_train[:, :, n])

    for n in range(0, grayscale_val.shape[2], 1):
        nome = name + "{:04n}".format(n)
        cv2.imwrite(dest_dir + "val/img/" + nome + estenzione, grayscale_val[:, :, n])
        cv2.imwrite(dest_dir + "val/label/" + "label_" + nome + estenzione, label_val[:, :, n])


