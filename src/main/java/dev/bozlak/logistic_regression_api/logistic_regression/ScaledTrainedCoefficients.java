package dev.bozlak.logistic_regression_api.logistic_regression;

public final class ScaledTrainedCoefficients {

    public final Double[] scaledWeights;
    public final Double scaledBias;

    public ScaledTrainedCoefficients(Double[] scaledWeights, Double scaledBias) {
        this.scaledWeights = scaledWeights;
        this.scaledBias = scaledBias;
    }
}
