package dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func.without_feature_scaling;

import dev.bozlak.logistic_regression_api.logistic_regression.InsideFuncForSigmoidFunction;
import dev.bozlak.logistic_regression_api.logistic_regression.LogisticRegression;
import dev.bozlak.logistic_regression_api.logistic_regression.ScaledTrainedCoefficients;

public class LogisticRegressionUseLinearFunction extends LogisticRegression {

    private final InsideFuncForSigmoidFunction multivariateLinearFunction;
    private Double[] weights;
    private Double bias;

    /**
     * For Output array:
     * Let true === 1 class, false === 0 class
     */
    public LogisticRegressionUseLinearFunction(
            Double[][] inputs,
            Boolean[] outputs,
            Double[] initialWeights,
            Double initialBias
    ) {
        super(inputs, outputs);
//        super.areWeightsCountEqualN(initialWeights);
        this.weights = initialWeights;
        this.bias = initialBias;

        multivariateLinearFunction = (oneSampleInputArray, weights, bias) -> {
            double result = 0.;
            final int n = super.getN();
            for (int j = 0; j < n; j++)
                result += weights[j] * oneSampleInputArray[j];

            return result + bias;
        };
    }

    public final ScaledTrainedCoefficients trainModelWithGradientDescent(double learningRate, int loopCount) {
        final int n = super.getN();
        Double[] tempCoefficients = new Double[n];

        for (int loopIndex = 0; loopIndex < loopCount ; loopIndex++) {
            for (int j = 0; j < n; j++) {
                tempCoefficients[j] = this.weights[j] - learningRate * this.dJ_dwj(j);
            }
            this.bias = this.bias - learningRate * this.dJ_db();

            System.arraycopy(tempCoefficients, 0, this.weights, 0, n);
        }

        return new ScaledTrainedCoefficients(this.weights, this.bias);
    }

    /**
     * @param j should be 0 <= j < n
     * @return dJ/dw_j partial derivative
     */
    private double dJ_dwj(int j) {
        this.validateIndexJ(j);

        return this.dJ_dwj_or_db(j);
    }

    private double dJ_db() {
        return this.dJ_dwj_or_db(super.getN());
    }

    public double sigmoidFunction (Double[] iThInputValues) {
        return this.sigmoidFunction(iThInputValues, this.weights, this.bias);
    }

    private double sigmoidFunction(Double[] iThInputValues, Double[] weights, Double bias) {
        return super.sigmoidFunction(this.multivariateLinearFunction, iThInputValues, weights, bias);
    }

    public double costFunction() {
        return this.costFunction(this.weights, this.bias);
    }

    public double costFunction(Double[] weights, double bias) {
        return super.costFunction(this.multivariateLinearFunction, weights, bias);
    }

    public byte convertBooleanToByte(boolean bool) {
        return bool ? (byte) 1 : (byte) 0;
    }

    private void validateIndexJ(int j) {
        if (j < 0 || j >= super.getN())
            throw new RuntimeException("index j should be 0 <= j < n !! Your j = " + j + " !!");
    }

    /**
     *
     * @param j For dJ_dwj j; For dJ_db j = n
     * @return dJ/dw_j or dJ/db
     */
    private double dJ_dwj_or_db(int j) {
        final int m = super.getM();
        double derivative = 0.;
        final int n = super.getN();

        double multiplier = 1.;
        for (int i = 0; i < m; i++) {
            if (j < n)
                multiplier = super.inputs[i][j];

            derivative +=
                    multiplier * (this.sigmoidFunction(super.inputs[i]) - this.convertBooleanToByte(super.outputs[i]));
        }
        return derivative / m;
    }

}