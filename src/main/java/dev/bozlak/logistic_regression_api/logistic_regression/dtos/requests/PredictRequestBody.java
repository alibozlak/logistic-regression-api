package dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests;

import dev.bozlak.logistic_regression_api.logistic_regression.ScaledTrainedCoefficients;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.ClientException;

public final class PredictRequestBody {

    public final ScaledTrainedCoefficients scaledTrainedCoefficients;
    public final Double[] input;

    public PredictRequestBody(ScaledTrainedCoefficients scaledTrainedCoefficients, Double[] input) {
        this.scaledTrainedCoefficients = scaledTrainedCoefficients;
        this.input = input;
    }

    public static Integer validatePredictRequestBody(
            ScaledTrainedCoefficients scaledTrainedCoefficients, Double[] input
    ) {
        final int n = scaledTrainedCoefficients.scaledWeights.length;

        if (n != input.length)
            throw new ClientException(
                    "Scaled Trained Coefficients's count (" + n + ") != input array size (" + input.length + ") !!"
            );

        return n;
    }
}
