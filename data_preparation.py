from pickle import load, dump
import tensorflow as tf
import numpy as np

def prep_entity_embedding_model(student_information):
        
    """
        input : ditionnary of student info 
        output: model predctions  
    """
    #load choice encoder
    encoder_c1 = load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_c1.pkl', 'rb'))
    encoder_c2 =  load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_c2.pkl', 'rb'))
    encoder_c3 =  load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_c3.pkl', 'rb'))
    encoder_c4 =  load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_c4.pkl', 'rb'))
    encoder_c5 =  load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_c5.pkl', 'rb'))
    encoder_c6 =  load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_c6.pkl', 'rb'))
    encoder_sexe= load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_SEXE.pkl', 'rb'))
    encoder_serie= load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_SERIE.pkl', 'rb'))
    encoder_wilaya_bac= load(open('orientationSystem\data_preparatio_object\categorical_encoder_entity_embedding_WILAYA_BAC.pkl', 'rb'))

    # load scaler
    scaler = load(open('orientationSystem\data_preparatio_object\scaler_marks_entity_embedding.pkl', 'rb'))
    sample= {
            'input_c1':encoder_c1.transform(np.array(student_information['c1']).reshape(-1, 1)),
            "input_c2":encoder_c2.transform(np.array(student_information['c2']).reshape(-1, 1)),
            "input_c3":encoder_c3.transform(np.array(student_information['c3']).reshape(-1, 1)),
            "input_c4":encoder_c4.transform(np.array(student_information['c4']).reshape(-1, 1)),
            "input_c5":encoder_c5.transform(np.array(student_information['c5']).reshape(-1, 1)),
            "input_c6":encoder_c6.transform(np.array(student_information['c6']).reshape(-1, 1)),
            "input_WILAYA_BAC":encoder_wilaya_bac.transform(np.array(int(student_information['wilaya_bac'])).reshape(-1, 1)),
            "input_SEXE":encoder_sexe.transform(np.array(student_information['sexe']).reshape(-1, 1)),
            "input_SERIE":encoder_serie.transform(np.array(student_information['serie_bac']).reshape(-1, 1)),
            "input_number_features": scaler.transform([[student_information['english'],
                                                       student_information['french'],
                                                       student_information['his_geo'],
                                                       student_information['arabic_literature'],
                                                       student_information['maths'],
                                                       student_information['philosophy'],
                                                       student_information['physics'],
                                                       student_information['primary_module'],
                                                       student_information['islamic_science'],
                                                       student_information['moyenne_bac']
                                                       ]])
            }
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    return sample
def create_student_dictionary(list_choice, etudiant):
    student_information = {
                'matricule':etudiant['matricule'],
                'moyenne_bac':etudiant['moyenne_bac'],
                'serie_bac':etudiant['serie_bac'],
                'wilaya_bac':etudiant['wilaya_bac'],
                'sexe':etudiant['sexe'],
                'english':etudiant['english'],
                'french':etudiant['french'],
                'his_geo':etudiant['his_geo'],
                'arabic_literature':etudiant['arabic_literature'],
                'maths':etudiant['maths'],
                'philosophy':etudiant['philosophy'],
                'physics':etudiant['physics'],
                'primary_module':etudiant['primary_module'],
                'islamic_science':etudiant['islamic_science'],
                'c1':list_choice[0],
                'c2':list_choice[1],
                'c3':list_choice[2],
                'c4':list_choice[3],
                'c5':list_choice[4],
                'c6':list_choice[5]
                    }
    return student_information