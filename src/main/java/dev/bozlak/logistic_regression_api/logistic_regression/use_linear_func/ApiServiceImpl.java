package dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func;

import dev.bozlak.logistic_regression_api.logistic_regression.ScaledTrainedCoefficients;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.RequestBody;
import dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses.SuccessResponseBody;
import dev.bozlak.logistic_regression_api.logistic_regression.feature_column_scaling.ScaledResult;
import dev.bozlak.logistic_regression_api.logistic_regression.feature_column_scaling.ScalingService;
import dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func.without_feature_scaling.LogisticRegressionUseLinearFunction;
import org.springframework.stereotype.Service;

@Service
public class ApiServiceImpl implements ApiService {

    private final ScalingService scalingService;

    public ApiServiceImpl(ScalingService scalingService) {
        this.scalingService = scalingService;
    }

    @Override
    public SuccessResponseBody trainModel(RequestBody requestBody) {
        ScaledResult scaledResult = this.scalingService.scale(requestBody.inputs);

        final int n = requestBody.inputs[0].length;
        Double[] initialWeights = new Double[n];
        for (int j = 0; j < n; j++)
            initialWeights[j] = 0.;

        LogisticRegressionUseLinearFunction logisticRegressionUseLinearFunction = new LogisticRegressionUseLinearFunction(
                scaledResult.scaledInputs, requestBody.outputs, initialWeights, 0.
        );
        Double J_beforeTrainWithScaleInputs = logisticRegressionUseLinearFunction.costFunction();

        ScaledTrainedCoefficients scaledTrainedCoefficients
                = logisticRegressionUseLinearFunction.trainModelWithGradientDescent(
                        requestBody.learningRate, requestBody.loopCount
        );
        Double J_afterTrainWithScaleInputs = logisticRegressionUseLinearFunction.costFunction();

        return new SuccessResponseBody(
                scaledTrainedCoefficients, J_beforeTrainWithScaleInputs, J_afterTrainWithScaleInputs
        );
    }
}
