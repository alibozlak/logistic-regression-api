package dev.bozlak.logistic_regression_api.logistic_regression;

@FunctionalInterface
public interface InsideFuncForSigmoidFunction {

    Double insideFuncForSigmoidFunc(Double[] oneInputValues, Double[] weights, Double bias);
}