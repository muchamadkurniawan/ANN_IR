import pandas as pd
from keras import Sequential
import tensorflow as tf
import Model
from Dataset import load_storage
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    load = load_storage()
    data_xest = load.x_test
    print(len(data_xest))
    print(data_xest[0].shape)

    ML = Model.ANNforImageClassfication(load.x_train, load.y_train, load.x_test, load.y_test)
    ML.creteModel()
    ML.compileModel()
    ML.ModelPredict(data_xest)
    print(load.y_test)
    ML.ModelEvaluate()

