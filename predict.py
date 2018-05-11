import numpy as np
from keras.models import load_model
from keras.preprocessing import image
classifier = load_model('my_model.h5')


test_image = image.load_img('C:/Users/JZHU54/PycharmProjects/Classification/mypic.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
classifier.summary
