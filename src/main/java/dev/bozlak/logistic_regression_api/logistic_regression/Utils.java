package dev.bozlak.logistic_regression_api.logistic_regression;

public final class Utils {

    /**
     * Java is OOP based Programming Language. So name = MULTIVARIATE_LINEAR_FUNCTION
     */
    public static InsideFuncForSigmoidFunction MULTIVARIATE_LINEAR_FUNCTION = (
            oneSampleInputArray, weights, bias
    ) -> {
        double result = 0.;
        final int n = oneSampleInputArray.length;
        for (int j = 0; j < n; j++)
            result += weights[j] * oneSampleInputArray[j];

        return result + bias;
    };

    public static double sigmoidFunction(
            InsideFuncForSigmoidFunction insideFuncForSigmoidFunction,
            Double[] inputArray,
            Double[] weights,
            Double bias
    ){
        return 1. / (1. + Math.pow(
                Math.E, -insideFuncForSigmoidFunction.insideFuncForSigmoidFunc(inputArray, weights, bias)
        ));
    }

    public static double sigmoidInsideLinearFunction(Double[] inputArray, Double[] weights, Double bias) {
        return sigmoidFunction(MULTIVARIATE_LINEAR_FUNCTION, inputArray, weights, bias);
    }

}
