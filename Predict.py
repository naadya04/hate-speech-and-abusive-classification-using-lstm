from Preprocessing import Preprocessing
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class Predict:
    
    def __init__(self):
        self.loaded_model = load_model('model_data/model/model.h5', compile=False)
        # self.loaded_model = load_model('model_data/model/model_best', compile=False)
        self.loaded_model.compile()
        self.token = 'model_data/token/tokenizer.pickle'
    
    def predictText(self,input):
        input_text = input 
        #preprocessing
        preprocessor = Preprocessing()
        input_text_preprocessed= preprocessor.preprocess(input_text)
        
        labels = ['HS', 'Abusive']

        # load model
        self.model = self.loaded_model

        # Get Prev Token    
        with open(self.token, 'rb') as file:
            tokenizer = pickle.load(file)

        # Melakukan tokenisasi dan padding pada input
        input_sequence = tokenizer.texts_to_sequences(input_text_preprocessed)
        input_padded_sequence = pad_sequences(input_sequence, maxlen=100)

        # Melakukan prediksi
        print('-------Melakukan Prediksi-------')
        prediction = self.model.predict(input_padded_sequence)[0]
        threshold = 0.5
        
        print(prediction)
        # Menampung hasil prediksi
        print("Hasil prediksi:")
        predict_result = []
        for i in range(len(labels)):
            if prediction[i] > threshold:
                predict_result.append(labels[i])
                
        return predict_result