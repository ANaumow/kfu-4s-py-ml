import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split as tts

os.chdir('c://task')

test_df = pd.read_csv('for_test')
train_df = pd.read_csv('for_train')

family = train_df[['family', 'genus', 'category_name']].groupby(['family', 'genus']).count()

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2)

m = train_df[['file_name', 'family', 'genus', 'category_id']]
fam = m.family.unique().tolist()
m.family = m.family.map(lambda x: fam.index(x))
gen = m.genus.unique().tolist()
m.genus = m.genus.map(lambda x: gen.index(x))

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)
train = train[:70000]
verif = verif[:10000]

shape = (50, 50, 3)
epochs = 2
batch_size = 32


def make_model(shape):
    input = Input(shape)
    cur_layer = Conv2D(3, (3, 3), activation='relu', padding='same')(input)
    cur_layer = Conv2D(3, (5, 5), activation='relu', padding='same')(cur_layer)
    cur_layer = MaxPool2D(pool_size=(3, 3), strides=(3, 3))(cur_layer)
    cur_layer = BatchNormalization()(cur_layer)
    cur_layer = Dropout(0.5)(cur_layer)
    cur_layer = Conv2D(16, (5, 5), activation='relu', padding='same')(cur_layer)
    cur_layer = MaxPool2D(pool_size=(5, 5), strides=(5, 5))(cur_layer)
    cur_layer = BatchNormalization()(cur_layer)
    cur_layer = Dropout(0.5)(cur_layer)
    cur_layer = Flatten()(cur_layer)

    o1 = Dense(310, name='f')(cur_layer)

    o2 = concatenate([o1, cur_layer])
    o2 = Dense(3678, name='g')(o2)
    o3 = concatenate([o1, o2, cur_layer])
    o3 = Dense(32094, name='c')(o3)
    cur_layer = Model(inputs=input, outputs=[o1, o2, o3])
    opt = Adam(lr=0.001, amsgrad=True)
    cur_layer.compile(optimizer=opt, loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy',
                                           'sparse_categorical_crossentropy'], metrics=['accuracy'])
    return cur_layer


model = make_model(shape)

for layers in model.layers:
    if layers.name == 'g' or layers.name == 'c':
        layers.trainable = False

model.fit_generator(train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='/trains/',
    x_col="file_name",
    y_col=["family", "genus", "category_id"],
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode='multi_output'),
    validation_data=train_datagen.flow_from_dataframe(
        dataframe=verif,
        directory='/trains/',
        x_col="file_name",
        y_col=["family", "genus", "category_id"],
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode='multi_output'),
    epochs=epochs,
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(verif) // batch_size,
    verbose=1,
    workers=8,
    use_multiprocessing=False)

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)
train = train[:70000]
verif = verif[:10000]

for layers in model.layers:
    if layers.name == 'g':
        layers.trainable = True

model.fit_generator(train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='/trains/',
    x_col="file_name",
    y_col=["family", "genus", "category_id"],
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode='multi_output'),
    validation_data=train_datagen.flow_from_dataframe(
        dataframe=verif,
        directory='/trains/',
        x_col="file_name",
        y_col=["family", "genus", "category_id"],
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode='multi_output'),
    epochs=epochs,
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(verif) // batch_size,
    verbose=1,
    workers=8,
    use_multiprocessing=False)

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)
train = train[:70000]
verif = verif[:10000]

for layers in model.layers:
    if layers.name == 'c':
        layers.trainable = True

model.fit_generator(train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='/trains/',
    x_col="file_name",
    y_col=["family", "genus", "category_id"],
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode='multi_output'),
    validation_data=train_datagen.flow_from_dataframe(
        dataframe=verif,
        directory='/trains/',
        x_col="file_name",
        y_col=["family", "genus", "category_id"],
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode='multi_output'),
    epochs=epochs,
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(verif) // batch_size,
    verbose=1,
    workers=8,
    use_multiprocessing=False)

model.save("my_super_model2")

batch_size = 32
test_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False)

prev_test_df = test_df.copy()

generator = test_datagen.flow_from_dataframe(
    dataframe=test_df.iloc[:50000],
    directory='/tests/',
    x_col='file_name',
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

family, genus, category = model.predict(generator, verbose=1)

sub = pd.DataFrame()
sub['Id'] = test_df.image_id
sub['Id'] = sub['Id'].astype('int32')
sub['Predicted'] = np.concatenate([np.argmax(category, axis=1), 7777 * np.ones((len(test_df.image_id) - len(category)))], axis=0)
sub['Predicted'] = sub['Predicted'].astype('int32')
sub.info()
sub = sub.sort_values('Id')
sub.to_csv('wwresult.txt', index=False)
