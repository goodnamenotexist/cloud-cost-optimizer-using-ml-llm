from ml_predictor import predict_optimization
from mini_llm import generate_suggestion

def analyze_cost(data):

    prediction = predict_optimization(data)

    if prediction == 1:
        label = "Inefficient"
    else:
        label = "Efficient"

    result = generate_suggestion(prediction, data)

    return result