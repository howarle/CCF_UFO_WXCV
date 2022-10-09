import logging
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
import keras.regularizers as regularizers
from keras.layers.core.embedding import Embedding
# from keras.layers.Embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import scipy.stats

from utils.quicksort import quicksort

class keras_train_model:

    name = None
    model = None
    model_acc = -1
    data_train_X = None
    data_train_Y = None
    data_test_X = None
    data_test_Y = None
    data_all_X = None
    data_all_Y = None
    trained = False

    def __init__(self) -> None:
        in_size = 24 * 2
        out_size = 2
        embedding_dim = 8
        model = Sequential()
        model.add(Embedding(50, embedding_dim, input_length=in_size))
        model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(out_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        # model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
        # # model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu', activity_regularizer='l1'))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu' ))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        # model.add(Dense(16, activation='relu'))
        # # model.add(Dense(128, activation='relu'))
        # model.add(Dense(out_size, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        self.model = model

    def conbime_data(self, Xi, input_X: list):
        a = [input_X[x] for x, y in Xi]
        b = [input_X[y] for x, y in Xi]
        X = np.concatenate((a,b), axis=1)
        # X = list(X)
        return X
        X = []
        for x, y in Xi:
            v = input_X[x] + input_X[y]
            X.append(v)
        return X

    def data_encode(self, input_X, input_Y = None):
        # X = []
        Y = []
        Xi = []
        input_X = [np.array(x) for x in input_X]
        for i1 in range(len(input_X)):
            for i2 in range(len(input_X)):
                if i1 != i2:
                    if input_Y is not None:
                        y = [1, 0] if (input_Y[i1] > input_Y[i2]) else [0, 1]
                        Y.append(y)
                    Xi.append([i1,i2])
        return self.conbime_data(Xi, input_X), np.array(Y), Xi

    def data_Y_decode(self, Y, Xi, x_size):
        score = [0 for i in range(x_size)]
        for i, y in enumerate(Y):
            i1, i2 = Xi[i]
            v = y[0] - y[1]
            score[i1] = score[i1] + v
            score[i2] = score[i2] - v
        return score

    def do_train_wrong(self, input_X, input_Y):
        X, Y, Xi = self.data_encode(input_X, input_Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

        self.data_train_X, self.data_train_Y, self.data_test_X, self.data_test_Y = X_train, X_test, y_train, y_test
        model = self.model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=2048, use_multiprocessing=True)

        y_test_pred = self.do_predict(input_X)
        self.model_acc = scipy.stats.kendalltau(y_test_pred, input_Y).correlation
        print(f"  -->  predict-acc = {self.model_acc}")

        return self.model_acc

    def do_train(self, input_X, input_Y):
        X_tr, X_te, y_tr, y_te = train_test_split(input_X, input_Y, test_size=.2)
        X_train, y_train, Xi = self.data_encode(X_tr, y_tr)
        X_test, y_test, Xi = self.data_encode(X_te, y_te)

        self.data_train_X, self.data_train_Y, self.data_test_X, self.data_test_Y = X_train, X_test, y_train, y_test
        model = self.model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=1024, use_multiprocessing=True)

        # train_pred = model.predict(X_train)
        # test_pred = model.predict(X_test)
        # print(f"train-acc = {accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))}")
        # print(f"test-acc = {accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))}")

        y_test_pred = self.do_predict(X_te)
        self.model_acc = scipy.stats.kendalltau(y_test_pred, y_te).correlation
        print(f"  -->  predict-acc = {self.model_acc}")

        return self.model_acc

    def do_predict(self, input_X):
        '''
        input_X: inout x axis
        sort with quick sort
        '''
        logging.info(f"start predict   input size:{len(input_X)}")
        if (False):
            X, _, Xi = self.data_encode(input_X)
            Y = self.model.predict(X, batch_size=4096, use_multiprocessing=True)
            out_Y = self.data_Y_decode(Y, Xi, len(input_X))
            return out_Y

        x_size = len(input_X)
        a = [i for i in range(x_size)]

        input_X = [np.array(x) for x in input_X]
        minimum = 200
        mg = quicksort(a, minimum)
        recod = {}
        cnt = 0
        while (True):
            cnt = cnt + 1
            tmp = []
            for it in mg.seg_list:
                if it.tot_siz > minimum:
                    tmp.append(it.tot_siz)
            qu_list = mg.query()
            print(f"cnt:{cnt}  len(recod):{len(recod)} len(ask_list):{len(qu_list)} len(mg.seg_list):{len(mg.seg_list)} {set(tmp)}")
            logging.debug(f"cnt:{cnt}  len(recod):{len(recod)} len(ask_list):{len(qu_list)} len(mg.seg_list):{len(mg.seg_list)} {set(tmp)}")

            pre_list = []
            for it in qu_list:
                if it not in recod:
                    pre_list.append(it)
            print("conbime_data...")
            logging.debug(f"conbime_data...")
            X = self.conbime_data(pre_list, input_X)
            logging.debug(f"conbime_data done")
            Y = self.model.predict(X, batch_size=4096, use_multiprocessing=True)
            
            for idx, it in enumerate(pre_list):
                recod[it] = float(Y[idx][0] - Y[idx][1])

            if mg.do(recod):
                break
        
        logging.info(f"finsh predict")
        ans = mg.ans
        assert(len(ans) == x_size)
        out_Y = [0 for i in range(x_size)]
        for i in range(len(ans)):
            out_Y[ans[i]] = i

        return out_Y

    def do_predict_slow(self, input_X):
        '''
        input_X: inout x axis
        sort with n^2 sort
        '''
        logging.info(f"start predict   input size:{len(input_X)}")
        x_size = len(input_X)
        input_X = [np.array(x) for x in input_X]

        gap = int(5e6/x_size)
        score = [0]*x_size

        for start_pos in range(0, x_size, gap):
            grp_sz = min(gap, x_size-start_pos)
            X = [np.concatenate([[input_X[i]]*x_size, input_X], axis = 1) for i in range(start_pos, start_pos+grp_sz)]
            X = np.concatenate(X, axis=0)
            Y = self.model.predict(X, batch_size=0, use_multiprocessing=True)
            X = None
            Y = Y.reshape(grp_sz, x_size, 2)
            for i in range(grp_sz):
                sc = np.sum(Y[i], axis=0)
                score[i+start_pos] = score[i+start_pos] + sc[0]-sc[1]
            Y = None


        # out_Y = self.data_Y_decode(Y, Xi, len(input_X))


        return score