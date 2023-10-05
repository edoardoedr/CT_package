import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import numpy as np
import cv2
import os
from PIL import Image, ImageFilter, ImageOps, ExifTags
import SimpleITK as sitk
import statistics as st
from scipy import ndimage, signal
import random
import time
from .utils import *


############################ INIZIO CLASSE ##############################################################

class CT_stack:
    def __init__(self, np_stack, voxel_size, tipology, name):
        # controlla se l'array è un array numpy
        if isinstance(np_stack, np.ndarray):
            self.np_stack = np_stack
        else:
            raise TypeError("Array must be a numpy array")

        # controlla se il voxel size è una tupla o una lista di tre elementi
        if isinstance(voxel_size, (tuple, list)) and len(voxel_size) == 3:
            self.voxel_size = voxel_size
        else:
            raise TypeError("Voxel size must be a tuple or a list of three elements")
        # controlla se il nome è una stringa

        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("Name must be a string")
        
        if isinstance(tipology, str):
            if tipology in "image,label,prediction,explained":
                self.tipology = tipology
            else:
                raise TypeError("tipology must be a string")
        else:
            raise TypeError("tipology must be a string")

    def print_info(self):
        print("le dimensioni dell'immagine sono: ", self.np_stack.shape)
        print("Il voxel ha dimensioni ", self.voxel_size)
        print("La tipologia dello stack è ", self.tipology)
        print("Il nome dello stack è ", self.name)

    @staticmethod
    def read_stack(data_dir, step):
        if os.path.isdir(data_dir):
            lista_immagini = os.listdir(data_dir)
            lista_immagini.sort()
            chars = list(data_dir)
            if chars[len(chars) - 1] == "/":
                chars.pop(len(chars) - 1)
                delimiter = ""
                data_dir = delimiter.join(chars)

            if not os.path.isfile(data_dir + "/" + lista_immagini[0]):
                print("La cartella selezionata non contiene solo immagini o non ne contiene")
                return None
            else:
                immagine_prova = Image.open(data_dir + "/" + lista_immagini[0])
                if immagine_prova.format == 'TIFF' and immagine_prova.mode == 'F':
                    numpy_type = np.float32
                elif immagine_prova.format == 'TIFF' and immagine_prova.mode == 'I':
                    numpy_type = np.uint32
                elif immagine_prova.format == 'TIFF' and (
                        immagine_prova.mode == 'I;16B' or immagine_prova.mode == 'I;16' or immagine_prova.mode == 'I;16L'):
                    numpy_type = np.uint16
                elif immagine_prova.format == 'TIFF' and (immagine_prova.mode == 'L' or immagine_prova.mode == 'P'):
                    numpy_type = np.uint8
                if data_dir.find("label") > 0:
                    tipology = "label"
                else:
                    tipology = "image"
                exif = immagine_prova.getexif()
                resolution = [1, 1, 1 * step]
                for tag_id in exif:
                    tag = ExifTags.TAGS.get(tag_id, tag_id)  # get the tag name or use the tag ID if not found
                    if tag == "XResolution":
                        value = exif.get(tag_id)  # get the tag value
                        resolution[0] = float(value)
                        resolution[2] = float(value) * step
                    elif tag == "YResolution":
                        value = exif.get(tag_id)  # get the tag value
                        resolution[1] = float(value)
                nome = data_dir.rsplit("/", 1)[1]
                nome = nome.rsplit(".", -1)[0]

                dimensioni = np.array(immagine_prova)
                righe = dimensioni.shape[0]
                colonne = dimensioni.shape[1]
                z = len(lista_immagini)
                zeta = int(z / step)
                stack = np.empty([righe, colonne, zeta], dtype=numpy_type)
                i = 0
                for n in range(0, z - step, step):
                    immagine = Image.open(data_dir + "/" + lista_immagini[n])
                    stack[:, :, i] = np.copy(immagine)
                    i = i + 1
            return CT_stack(stack, resolution, tipology, nome)

        elif os.path.isfile(data_dir):
            stack = sitk.ReadImage(data_dir)
            stack_np = sitk.GetArrayViewFromImage(stack)
            stack_np = np.transpose(stack_np, (1, 2, 0))
            stack_np = resample(stack_np, step)
            resolution_in = stack.GetSpacing()
            resolution = [resolution_in[1], resolution_in[2], resolution_in[0] * step]
            nome = data_dir.rsplit("/", 1)[1]
            nome = nome.rsplit(".", -1)[0]
            if data_dir.find("label") > 0:
                tipology = "label"
            else:
                tipology = "image"
            return CT_stack(stack_np, resolution, tipology, nome)

    def resize_stack(self, new_size = None, scale = None):

        stack = sitk.GetImageFromArray(self.np_stack)
        orig_size = stack.GetSize() #x,y,z
        orig_spacing = self.voxel_size

        if new_size is not None:
            assert len(new_size) == 3
            for s in [0,1,2]:
                if new_size[s] == "o":
                    new_size[s] = orig_size[s]
        
        elif scale is not None:
            assert len(scale) == 3
            new_size = [int(d * s) for d,s in zip(orig_size, scale)]
        
        else:
            print("Un parametro tra new_size o scale deve essere non nullo")


        # Define the new size and spacing
        new_spacing = [osz * osp / nsz for osz, osp, nsz in zip(orig_size, orig_spacing, new_size)]

        # Create a resample filter with default parameters
        resize = sitk.ResampleImageFilter()

        # Set the desired output parameters
        resize.SetSize(new_size)
        resize.SetOutputSpacing(new_spacing)
        resize.SetOutputOrigin(stack.GetOrigin())
        resize.SetOutputDirection(stack.GetDirection())

        # Set the interpolator to linear (default is nearest neighbor)
        if self.tipology == "label" or self.tipology == "prediction":
            resize.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resize.SetInterpolator(sitk.sitkLinear)
        # Resample the image
        resized_stack = resize.Execute(stack)
        resized_stack_np = np.copy(sitk.GetArrayViewFromImage(resized_stack))
        resized_stack_np = np.transpose(resized_stack_np, (1, 2, 0))
        return CT_stack(resized_stack_np, new_spacing, self.tipology, self.name)

    def save_stack(self, dest_dir, estenzione):

        if estenzione == "tif" or estenzione == ".tif":
            z = self.shape[2]
            if not os.path.exists(dest_dir + self.name + "/"):
                os.mkdir(dest_dir + self.name)
            for n in range(0, z, 1):
                nome = self.name + "/" + self.name + "{:04n}".format(n)
                cv2.imwrite(dest_dir + nome + ".tif", self.np_stack[:, :, n])
        elif estenzione == "png" or estenzione == ".png":
            z = self.shape[2]
            if not os.path.exists(dest_dir + self.name + "/"):
                os.mkdir(dest_dir + self.name)
            for n in range(0, z, 1):
                nome = self.name + "/" + self.name + "{:04n}".format(n)
                cv2.imwrite(dest_dir + nome + ".png", self.np_stack[:, :, n])
        else:
            stack_tosave = np.transpose(self.np_stack, (2, 0, 1))
            stack_sitk = sitk.GetImageFromArray(stack_tosave)
            spaziatura = [self.voxel_size[2], self.voxel_size[0], self.voxel_size[1]]
            stack_sitk.SetSpacing(spaziatura)
            if self.tipology == "image":
                tipologia = ""
                nome = self.name
            elif self.tipology == "label" and "label" in self.name:
                tipologia = ""
                nome = self.name
            elif self.tipology == "label" and "label" not in self.name:
                tipologia = "_label"
                nome = self.name
            elif self.tipology == "prediction":
                tipologia = "_prediction"
                nome = self.name
                if "label" in nome:
                    nome = nome.replace("_label","")
            elif self.tipology == "explained":
                tipologia = "_explained"
                nome = self.name
                if "label" in nome:
                    nome = nome.replace("_label","")

            sitk.WriteImage(stack_sitk, dest_dir + nome + tipologia + estenzione)
        print("Stack saved")

    def crop_ROI(self, coordinate=[]):

        if len(coordinate) > 1:
            self.np_stack = crop_stack(self.np_stack, coordinate)
        else:
            coordinate = where_crop_ROI(self.np_stack)
            self.np_stack = crop_stack(self.np_stack, coordinate)

    @staticmethod
    def multiply(stack_one, stack_two):
        assert isinstance(stack_one, CT_stack), "Le immagini devono essere di tipo stack images"
        assert isinstance(stack_two, CT_stack), "Le immagini devono essere di tipo stack images"

        new_stack_array = stack_one.np_stack * stack_two.np_stack
        nuovo_nome = stack_one.name
        nuovo_voxel_size = stack_one.voxel_size
        nuovo_typology = stack_one.tipology

        return CT_stack(new_stack_array, nuovo_voxel_size, nuovo_typology, nuovo_nome)

    def copy_stack(self):
        new_stack = np.copy(self.np_stack)

        return CT_stack(new_stack, self.voxel_size, self.tipology, self.name)


    def mode_filter_25D(self, filter_size):

        start_time = time.perf_counter()
        rig = self.np_stack.shape[0]
        col = self.np_stack.shape[1]
        zeta = self.np_stack.shape[2]

        filtro = partial(filtro_mode, filter_size)

        # filtriamo su xy
        list_stack = []
        for n in range(zeta):
            list_stack.append(self.np_stack[:, :, n])

        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(zeta):
            self.np_stack[:, :, n] = np.copy(result[n])

        # filtriamo su xz
        list_stack = []
        for n in range(rig):
            list_stack.append(self.np_stack[n, :, :])
        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(rig):
            self.np_stack[n, :, :] = np.copy(result[n])

        finish_time = time.perf_counter()
        print("Filtro mode finito in {} seconds - using multiprocessing".format(finish_time - start_time))
        print("---")

        return CT_stack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def median_filter_25D(self, filter_size):

        start_time = time.perf_counter()
        rig = self.np_stack.shape[0]
        col = self.np_stack.shape[1]
        zeta = self.np_stack.shape[2]

        filtro = partial(signal.medfilt2d, kernel_size=filter_size)

        # filtriamo su xy
        list_stack = []
        for n in range(zeta):
            list_stack.append(self.np_stack[:, :, n])

        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(zeta):
            self.np_stack[:, :, n] = np.copy(result[n])

        # filtriamo su xz
        list_stack = []
        for n in range(rig):
            list_stack.append(self.np_stack[n, :, :])
        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(rig):
            self.np_stack[n, :, :] = np.copy(result[n])

        finish_time = time.perf_counter()
        print("Filtro median finito in {} seconds - using multiprocessing".format(finish_time - start_time))
        print("---")

        return CT_stack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def unsharpmask(self):

        start_time = time.perf_counter()
        rig = self.np_stack.shape[0]
        col = self.np_stack.shape[1]
        zeta = self.np_stack.shape[2]

        filtro = FiltroUnsharp

        # filtriamo su xy
        list_stack = []
        for n in range(zeta):
            list_stack.append(self.np_stack[:, :, n])

        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(zeta):
            self.np_stack[:, :, n] = np.copy(result[n])

        if self.np_stack.dtype == np.float32:
            massimo = self.np_stack.max()
            minimo = self.np_stack.min()
            new_array = (self.np_stack - minimo)/(massimo - minimo)
            self.np_stack = new_array

        finish_time = time.perf_counter()
        print("Filtro unsharpmask finito in {} seconds - using multiprocessing".format(finish_time - start_time))
        print("---")

        return CT_stack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def fill_holes(self, label_riempire):

        start_time = time.perf_counter()
        rig = self.np_stack.shape[0]
        col = self.np_stack.shape[1]
        zeta = self.np_stack.shape[2]
        struttura = np.ones((5, 5))

        filtro = partial(fill_holed2D, label_riempire, struttura)

        # filtriamo su xy
        list_stack = []
        for n in range(zeta):
            list_stack.append(self.np_stack[:, :, n])

        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(zeta):
            self.np_stack[:, :, n] = np.copy(result[n])

        # filtriamo su xz
        list_stack = []
        for n in range(rig):
            list_stack.append(self.np_stack[n, :, :])
        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(rig):
            self.np_stack[n, :, :] = np.copy(result[n])

        finish_time = time.perf_counter()
        print("Filtro fill holes finito in {} seconds - using multiprocessing".format(finish_time - start_time))
        print("---")

        return CT_stack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def sharpen(self):

        start_time = time.perf_counter()
        rig = self.np_stack.shape[0]
        col = self.np_stack.shape[1]
        zeta = self.np_stack.shape[2]

        filtro = Filtrosharpen

        # filtriamo su xy
        list_stack = []
        for n in range(zeta):
            list_stack.append(self.np_stack[:, :, n])

        with Pool() as pool:
            result = pool.map(filtro, list_stack)

        for n in range(zeta):
            self.np_stack[:, :, n] = np.copy(result[n])

        if self.np_stack.dtype == np.float32:
            minimo = self.np_stack.min()
            self.np_stack[self.np_stack == 0] = minimo
            massimo = self.np_stack.max()
            new_array = (self.np_stack - minimo)/(massimo - minimo)
            self.np_stack = new_array

        finish_time = time.perf_counter()
        print("Filtro unsharpmask finito in {} seconds - using multiprocessing".format(finish_time - start_time))
        print("---")

        return CT_stack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def change_values(self, valori, substitute):
        if len(valori) == 2 and valori[1] == "=":
            self.np_stack[self.np_stack == valori[0]] = substitute
        elif len(valori) == 2 and valori[1] == ">":
            self.np_stack[self.np_stack > valori[0]] = substitute
        elif len(valori) == 2 and valori[1] == "<":
            self.np_stack[self.np_stack < valori[0]] = substitute
        elif len(valori) == 3 and valori[1] == "<>":
            self.np_stack[(self.np_stack < valori[0]) | (self.np_stack > valori[2])] = substitute
        elif len(valori) == 3 and valori[1] == "><":
            self.np_stack[(self.np_stack > valori[0]) and (self.np_stack < valori[2])] = substitute
        return CT_stack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def convert(self, type):
        if type == "8bit":
            stack_8bit = np.copy(self.np_stack).astype(np.float32)
            stack_8bit[stack_8bit < 0] = 0
            stack_8bit = (stack_8bit/stack_8bit.max())*255
            stack_8bit = np.round(stack_8bit).astype(np.uint8)
            return CT_stack(stack_8bit, self.voxel_size, self.tipology, self.name)
        elif type == "16bit":
            stack_16bit = np.copy(self.np_stack).astype(np.float32)
            stack_16bit[stack_16bit < 0] = 0
            stack_16bit = (stack_16bit/stack_16bit.max())*255
            stack_16bit = np.round(stack_16bit).astype(np.uint16)
            return CT_stack(stack_16bit, self.voxel_size, self.tipology, self.name)
        elif type == "normalize":
            stack_01 = np.copy(self.np_stack).astype(np.float32)
            stack_01[stack_01 < 0] = 0
            stack_01 = stack_01/stack_01.max()
            return CT_stack(stack_01, self.voxel_size, self.tipology, self.name)

