import numpy as np

# import the required layers from keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

def build_training_model():
    # create the CNN model
    model = Sequential()

    # step1: add the conv layer
    model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

    # step2: pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # additional layers
    # model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # step3: Flattening (input)
    model.add(Flatten())

    # step4: Full Connection (ANN)
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    # softmax, softplus

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # non-binary = categorical_crossentropy

    # print(model.summary())

    # reading the images

    from keras.preprocessing.image import ImageDataGenerator
    generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # test_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # get the training images
    training_set = generator.flow_from_directory('../images/training_set', target_size=(64, 64), class_mode='binary', batch_size=32)
    testing_set = generator.flow_from_directory('../images/test_set', target_size=(64, 64), class_mode='binary', batch_size=32)

    # git the data
    model.fit_generator(training_set, validation_data=testing_set, epochs=1)

    # save the model
    with open('model.json', 'w') as file:
        file.write(model.to_json())

    # save the weights
    model.save('model.h5')


def test_model(model):
    from keras.preprocessing.image import load_img, img_to_array

    image = load_img('cat_or_dog_1.jpg')
    array = img_to_array(image)
    resized_image = np.resize(array, (64, 64, 3))
    input_array = resized_image[np.newaxis, ...]
    type = model.predict(input_array)
    print(f"type = {type}")

    if (type[0][0] == 0.0):
        print("this is a cat")
    else:
        print("this is a dog")


if __name__ == '__main__':
    # build_training_model()

    from keras.models import model_from_json
    file = open('model.json', 'r')
    data = file.read()

    # read the model
    model = model_from_json(data)

    # apply the weights
    model.load_weights('model.h5')

    file.close()

    test_model(model)



























