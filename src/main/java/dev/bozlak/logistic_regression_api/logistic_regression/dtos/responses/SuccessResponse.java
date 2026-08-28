package dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses;

import dev.bozlak.logistic_regression_api.logistic_regression.TrainedCoefficients;

public class SuccessResponse {

    public final TrainedCoefficients trainedCoefficients;
    public final Double J_beforeTrain;
    public final Double J_afterTrain;

    public SuccessResponse(TrainedCoefficients trainedCoefficients, Double j_beforeTrain, Double j_afterTrain) {
        this.trainedCoefficients = trainedCoefficients;
        J_beforeTrain = j_beforeTrain;
        J_afterTrain = j_afterTrain;
    }
}
