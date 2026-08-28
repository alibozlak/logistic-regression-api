package dev.bozlak.logistic_regression_api.logistic_regression.feature_column_scaling;

public final class ScaledResult {

    public final Double[][] scaledInputs;
    public final Integer[] scaleRatiosPower10;

    public ScaledResult(Double[][] scaledInputs, Integer[] scaleRatiosPower10) {
        this.scaledInputs = scaledInputs;
        this.scaleRatiosPower10 = scaleRatiosPower10;
    }

}
