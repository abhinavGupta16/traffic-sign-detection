import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import os

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

data_path = "/scratch/ag7387/cv/Assignment2/data2/train_images/"

# our folder path containing some images
for class_name in os.listdir(data_path):
    print(class_name)
    folder_path = data_path + class_name
    # the number of file to generate
    num_files_desired = 2

    # loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    num_generated_files = 0
    for img in images:
        if num_generated_files >= num_files_desired:
            break
        # random image from the folder
        image_path = img
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)




        # dictionary of the transformations functions we defined earlier
        available_transformations = {
            'rotate': random_rotation,
            'horizontal_flip': horizontal_flip
        }

        # random num of transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1



        # define a name for our new file
        new_file_path = '%s/augmented_image_%s.ppm' % (folder_path, num_generated_files)

        # write image to the disk
        sk.io.imsave(new_file_path, transformed_image)
        num_generated_files+=1
