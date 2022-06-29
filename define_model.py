# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import scipy
import h5py

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	#opt = SGD(lr=0.01, momentum=0.99)
	model.summary()
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.subplots_adjust(wspace=0.3)
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_accuracy.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generators
	train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',class_mode='binary', batch_size=64, target_size=(200, 200))
	val_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/val/', class_mode='binary', batch_size=64, target_size=(200, 200))

	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=val_it, validation_steps=len(val_it), epochs=20, verbose=0)
	# evaluate model
	_, accTrain = model.evaluate(train_it, steps=len(train_it), verbose=0)
	_, accVal = model.evaluate(val_it, steps=len(val_it), verbose=0)
	_, accTest = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('Train: > %.3f' % (accTrain * 100.0))
	print('Val: > %.3f' % (accVal * 100.0))
	print('Test: > %.3f' % (accTest * 100.0))
	
	# learning curves
	summarize_diagnostics(history)
	model.save('end_model.h5')

# entry point, run the test harness
if __name__ == "__main__":
	run_test_harness()