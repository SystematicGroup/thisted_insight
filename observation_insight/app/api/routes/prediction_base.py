def predict(thisted_state, text):
    prediction_ = thisted_state.prediction_

    data_features = prediction_.preprocessing(text, thisted_state.encoder, thisted_state.nlp)
    result_scheme, result_prob = prediction_.predict_obs_scheme(data_features, thisted_state.clf, thisted_state.label_encoder, thisted_state.label_encoder_model)
    results = prediction_.get_results(result_scheme, result_prob)
    results = prediction_.medicin_fix(results)
    return results