from PIL import Image

def blend_image_arr(image1_arr, image2_arr, size=(224, 224), alpha=0.5):
    # convert ke objetc PIL.Image
    image1_PIL = Image.fromarray(image1_arr).resize(size)
    image2_PIL = Image.fromarray(image2_arr).resize(size)

    blended_image_PIL = Image.blend(image1_PIL, image2_PIL, alpha)
    
    return blended_image_PIL