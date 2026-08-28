package dev.bozlak.logistic_regression_api.logistic_regression.feature_column_scaling;

public interface ScalingService {

    ScaledResult scale(Double[][] requestInputs);
}
