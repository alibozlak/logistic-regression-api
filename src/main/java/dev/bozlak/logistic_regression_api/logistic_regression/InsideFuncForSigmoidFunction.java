package dev.bozlak.logistic_regression_api.logistic_regression;

@FunctionalInterface
public interface InsideFuncForSigmoidFunction {

    double insideFuncForSigmoidFunc(double[] oneInputValues, double[] weights, double bias);
}