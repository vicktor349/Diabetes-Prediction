import numpy as np
import pickle



loaded_model = pickle.load(open('C:/Users/ayomi/Documents/programming/diabetes_prediction/trained_model.sav', 'rb'))
input_data = (4,132,0,0,0,32.9,0.302,23)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print("This patient is not diabetic")
else:
  print("This patient is diabetic")