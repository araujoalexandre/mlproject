"""
__file__

    entity_embedding.py

__description__

    Model Class for Entity Embedding
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding

from .base import Wrapper
from kaggle.utils.functions import make_directory


class EntityEmbedding(Wrapper):
    

    def __init__(self, params, paths):        

        self.name = 'EntityEmbedding'

        self.nb_epoch = 10
        self._build_keras_model()

        super(EntityEmbedding, self).__init__(params, paths)


    def preprocessing(X):
        """
            xx
        """
        X_list = []


        return X_list


    def _build_model(self):
        """
            xxx
        """
        models = []

        binary_features = Sequential()
        binary_features.add(Dense(99, input_dim=1))
        models.append(binary_features)

        for (x, y) in [ ... ]:

            y = x // 2

            seq = Sequential()
            seq.add(Embedding(x, y, input_length=1))
            seq.add(Reshape(target_shape=(y,)))
            models.append(seq)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')


    def train(self, X_train, X_cv, y_train, y_cv):
        """
            xxx
        """
        X_train = self.split_features(X_train)
        X_cv = self.split_features(X_cv)

        self.model.fit(X_train, y_train, 
                        validation_data=(X_cv, y_cv),
                        nb_epoch=self.nb_epoch, 
                        batch_size=128)


    def predict(self, X):
        """
            function to make and return prediction
        """
        X = self.preprocessing(X)
        predict = self.model.predict(X).flatten()
        return predict