package dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func;

import dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.PredictRequestBody;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.RequestBodyForTrain;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses.SuccessResponseBody;

public interface ApiService {

    SuccessResponseBody trainModel(RequestBodyForTrain requestBodyForTrain);

    Double predict(PredictRequestBody predictRequestBody, Integer n);

}
