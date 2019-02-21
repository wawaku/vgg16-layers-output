import numpy as np
import cv2
import os
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


pic_folder = "./img/"
vgg_layers_folder = "./vgg_layers/"
model_vgg = VGG16(weights='imagenet')
model_vgg.summary()
list_name = os.listdir(pic_folder)

for i, file_name in enumerate(list_name):
    img = load_image(pic_folder + file_name)
    predictions = model_vgg.predict(img)
    top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    b_out_put = 0
    for inter_layer in model_vgg.layers:
        b_out_put += 1
        if b_out_put == 1 or (b_out_put == len(model_vgg.layers) - 3):
            continue
        get_output_function = K.function([model_vgg.input], [inter_layer.output])
        layer_output = get_output_function([img])[0]
        print(layer_output.shape)
        layer_output_cons = np.split(layer_output, layer_output.shape[3], 3)
        for j, con in enumerate(layer_output_cons):
            print(con.shape)
            con_img = np.reshape(con,(con.shape[1], con.shape[2], con.shape[0]))
            print(con_img.shape)
            con_img = 255* (con_img / np.max(con_img))
            print(con_img)
            os.makedirs(vgg_layers_folder + inter_layer.name, exist_ok=True)
            cv2.imwrite(vgg_layers_folder + inter_layer.name + "/" + str(j) + ".jpg", con_img)

