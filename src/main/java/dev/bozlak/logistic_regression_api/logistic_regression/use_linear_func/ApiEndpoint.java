package dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func;

import dev.bozlak.logistic_regression_api.logistic_regression.ScaledTrainedCoefficients;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.PredictRequestBody;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.RequestBodyForTrain;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses.SuccessResponseBody;
import org.springframework.aot.hint.annotation.RegisterReflectionForBinding;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/linear-regression/use-linear-func")
// Spring reaches the request and response bodies through this controller during
// the AOT pass and registers them for reflection by walking their properties.
// These DTOs carry public fields instead of getters, so the walk stops at the
// outermost type and ScaledTrainedCoefficients is never reached: in a native
// image it serialises as {} and deserialises to null. The JVM is unaffected.
@RegisterReflectionForBinding(ScaledTrainedCoefficients.class)
public class ApiEndpoint {

    private final ApiService apiService;

    public ApiEndpoint(ApiService apiService) {
        this.apiService = apiService;
    }

    @PostMapping("/train")
    public ResponseEntity<SuccessResponseBody> trainModel(
            @RequestBody RequestBodyForTrain requestBodyForTrain
    ) {
        RequestBodyForTrain.validateRequestBody(
                requestBodyForTrain.inputs,
                requestBodyForTrain.outputs,
                requestBodyForTrain.learningRate,
                requestBodyForTrain.loopCount
        );

        return new ResponseEntity<>(
                this.apiService.trainModel(requestBodyForTrain),
                HttpStatus.OK
        );
    }

    @PostMapping("/predict")
    public ResponseEntity<Double> predict(@RequestBody PredictRequestBody predictRequestBody) {
        int n = PredictRequestBody.validatePredictRequestBody(
                predictRequestBody.scaledTrainedCoefficients, predictRequestBody.input
        );

        return new ResponseEntity<>(
                this.apiService.predict(predictRequestBody, n),
                HttpStatus.OK
        );
    }

}
