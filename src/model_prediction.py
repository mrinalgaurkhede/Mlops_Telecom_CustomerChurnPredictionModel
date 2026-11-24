

def model_predict(model,X_test_transform):
    y_pred = model.predict(X_test_transform) # Unseen data

    return y_pred