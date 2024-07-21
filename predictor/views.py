from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Get the model filename from the request
            model_filename = request.POST.get('diabetes_model.pkl')

            # # Load the model and scaler from the provided filenames
            model = joblib.load('diabetes_model.pkl')
            # scaler_filename = model_filename.replace('diabetes_model.pkl', 'diabetes_scaler.pkl')
            scaler = joblib.load('diabetes_scaler.pkl')


            # Get the input features from the request
            data = request.POST
            features = [
                float(data['pregnancies']),
                float(data['glucose']),
                float(data['bloodpressure']),
                float(data['skinthickness']),
                float(data['insulin']),
                float(data['bmi']),
                float(data['dpf']),
                float(data['age'])
            ]
            features = np.array(features).reshape(1, -1)

            # Scale the input features
            scaled_features = scaler.transform(features)

            # Make the prediction using the loaded model
            prediction = model.predict(scaled_features)[0]
            result = 'Diabetic' if prediction == 1 else 'Not Diabetic'

            # Load the training data (replace 'X_train.pkl' with your actual file)
            X_train = joblib.load('X_train.pkl')

            # Create a LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train, 
                feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                class_names=['No Diabetes', 'Diabetes'],
                mode='classification'
            )

            # Explain the prediction using LIME
            explanation = explainer.explain_instance(
                scaled_features[0],  
                model.predict_proba,
                num_features=5
            )

            # Generate the LIME graph
            fig = explanation.as_pyplot_figure()  # Remove the 'ax' argument
            plt.close(fig)  # Close the figure to avoid memory leaks

            # Convert the plot to a base64 encoded image
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            return JsonResponse({'result': result, 'lime_graph': image_base64})
        except FileNotFoundError:
            return JsonResponse({'result': 'Error: Model file not found.'})
        except Exception as e:
            return JsonResponse({'result': f'Error test: {str(e)}'})
    return JsonResponse({'result': 'Error: Invalid request method.'})