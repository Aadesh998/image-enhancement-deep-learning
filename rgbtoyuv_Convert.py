import cv2

def rgb_to_yuv(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    cv2.imwrite('output_yuv.png', img_yuv)
    
    return img_yuv


image_path = '/home/aadesh-kumar/Documents/personal_project/isro_code/data/frames/moon_0000.png'
rgb_to_yuv(image_path)
