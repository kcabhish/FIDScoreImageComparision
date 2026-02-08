from CalculateFID import load_and_preprocess_images, calculate_fid
from tensorflow.keras.applications.inception_v3 import InceptionV3
# Paths to your dataset folders
real_images_path = 'real_images/'
generated_images_path = 'fake_images/'

def main():
    # Load and preprocess images
    realImages = load_and_preprocess_images(real_images_path)
    fakeImages = load_and_preprocess_images(generated_images_path)
    inceptionModel = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Calculate FID
    fid = calculate_fid(inceptionModel,realImages, fakeImages)
    print(f"FID: {fid}")


if __name__ == "__main__":
    main()
