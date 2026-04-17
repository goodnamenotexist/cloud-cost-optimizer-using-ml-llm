from ml_predictor import predict_optimization
from mini_llm import generate_suggestion

def analyze_cost(data):

    # Step 1: ML Prediction
    prediction = predict_optimization(data)

    # Step 2: Convert numeric output to label (optional)
    if prediction == 1:
        label = "Inefficient"
    else:
        label = "Efficient"

    # Step 3: Pass to Mini LLM
    result = generate_suggestion(prediction, data)

    return result