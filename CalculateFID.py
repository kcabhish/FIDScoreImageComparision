import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage import io
from scipy.linalg import sqrtm

def load_and_preprocess_images(directory, target_size=(299, 299)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adjust if different image format
            image = io.imread(os.path.join(directory, filename))
            if image.shape[-1] == 4:
                image = image[..., :3]  # Remove alpha channel if present
            image = tf.image.resize(image, target_size)
            image = preprocess_input(image.numpy())
            images.append(image)
    return np.array(images)

def calculate_fid(model, images1, images2):
    # Calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid