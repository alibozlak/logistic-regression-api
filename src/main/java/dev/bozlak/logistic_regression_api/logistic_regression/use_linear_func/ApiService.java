package dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func;

import dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.RequestBody;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses.SuccessResponseBody;

public interface ApiService {

    SuccessResponseBody trainModel(RequestBody requestBody);

}
