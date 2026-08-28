package dev.bozlak.logistic_regression_api.logistic_regression.exceptions;

public class WeightArraySizeandFeatureCountMismatchException extends ClientException {
    public WeightArraySizeandFeatureCountMismatchException(int n, int weightsSize) {
        super(
                "Weight array size (" + weightsSize + ") != Feature size (" + n + ") !!"
        );
    }
}