package dev.bozlak.logistic_regression_api.logistic_regression;

public final class TrainedCoefficients {

    public final double[] weights;
    public final double bias;

    public TrainedCoefficients(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }
}
