import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import layers
import numpy as np
def entity_embedding():
    """
        construct the model of the method entity embedding
    """
    num_features = ['english','french','his_geo','arabic_literature','maths','philosophy','physics',
                    'primary_module','islamic_science','MOYENNE_BAC']
            
    cat_features = ['c1','c2','c3','c4','c5','c6','WILAYA_BAC','SEXE','SERIE']
    inputs = []
    models =[]
    data = pd.read_csv('orientationSystem\data\dataset.csv')
    CLASSES = data['target'].nunique()

    dic_rate_dropout_embedding={
    'rate_embc1':0.22284237228700002,
    'rate_embc2': 0.4850945229846516,
    'rate_embc3':0.41207625179489626,
    'rate_embc4': 0.3219710066167931,
    'rate_embc5':0.17144074450780816,
    'rate_embc6':0.2770434364897959,
    'rate_embWILAYA_BAC':0.25530852093544626,
    'rate_embSEXE':0.3060378648324149,
    'rate_embSERIE':0.20079302042078964
    }
    for c in cat_features:
        
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2),50))
        inp = layers.Input(shape=(1,),name='input_'+'_'.join(c.split(' ')))
        inputs.append(inp)
        embed = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        embed= layers.SpatialDropout1D(rate=dic_rate_dropout_embedding["rate_emb{}".format(c)])(embed)
        embed_reshaped = layers.Reshape(target_shape=(embed_dim, ))(embed)
        models.append(embed_reshaped)
    num_input = tf.keras.layers.Input(shape=(len(num_features)),name='input_number_features')
    inputs.append(num_input)
    models.append(num_input)
    
    dict_num_hidden={
        'n_units_l0':222,
        'n_units_l1':411,
    }
    dict_drop_out={
    'rate0':0.1128020450526905,
    'rate1':0.2275167709610226
    }
    merge_models= tf.keras.layers.concatenate(models)
    for i in range(2):
            #optimum number of hidden nodes
        num_hidden =  dict_num_hidden["n_units_l{}".format(i)]
            #optimum activation function
        merge_models = tf.keras.layers.Dense(
                num_hidden,
                activation="relu",
                )(merge_models)
        merge_models = tf.keras.layers.BatchNormalization()(merge_models)
        merge_models =  tf.keras.layers.Dropout(rate=dict_drop_out["rate{}".format(i)])(merge_models)
    


    pred=tf.keras.layers.Dense(CLASSES,activation=tf.keras.activations.softmax)(merge_models)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5,min_lr=1e-08,verbose=1)
    
    model_full = tf.keras.models.Model(inputs= inputs,\
                                        outputs =pred)
    model_full.compile(loss='categorical_crossentropy',metrics=["accuracy"
                                                            ,tfa.metrics.F1Score(
                                                            num_classes=CLASSES,
                                                            average='macro',
                                                            name='f1_score_macro'),
                                                            tfa.metrics.F1Score(
                                                            num_classes=CLASSES,
                                                            average='weighted',
                                                            name='f1_score_weighted')], optimizer='Adam')
    model_full.load_weights("orientationSystem\models_weights\model_weights_entity_embedding.h5")
    return model_full