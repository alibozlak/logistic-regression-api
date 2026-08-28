package dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses;

import dev.bozlak.logistic_regression_api.logistic_regression.ScaledTrainedCoefficients;

public class SuccessResponseBody {

    public final ScaledTrainedCoefficients scaledTrainedCoefficients;
    public final Double J_beforeTrainScaled;
    public final Double J_afterTrainScaled;

    public SuccessResponseBody(
            ScaledTrainedCoefficients scaledTrainedCoefficients,
            Double j_beforeTrainScaled,
            Double j_afterTrainScaled
    ) {
        this.scaledTrainedCoefficients = scaledTrainedCoefficients;
        J_beforeTrainScaled = j_beforeTrainScaled;
        J_afterTrainScaled = j_afterTrainScaled;
    }
}
