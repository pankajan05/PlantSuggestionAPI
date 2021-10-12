from django.http import HttpResponse
from rest_framework.utils import json
from rest_framework.views import APIView
from keras.models import load_model
import numpy as np

d = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
     'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
     'rice', 'watermelon']


class plant(APIView):
    def post(self, request):
        if request.method == 'POST':
            print(request.data)

        model1 = load_model('./model/my_model.h5')
        response = d[np.argmax(model1.predict([[request.data['N'], request.data['P'], request.data['K'], request.data['temperature'], request.data['humidity'], request.data['ph'], request.data['rainfall']]]))]
        return HttpResponse(json.dumps({"result":response}),'application/json')


