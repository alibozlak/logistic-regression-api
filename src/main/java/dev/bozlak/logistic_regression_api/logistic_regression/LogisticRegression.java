package dev.bozlak.logistic_regression_api.logistic_regression;

import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.EachInputDatasNotSameSizeException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.InputAndOutputCountMismatchException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.WeightArraySizeandFeatureCountMismatchException;

public class LogisticRegression {

    protected final Double[][] inputs;

    /**
     * For Output array:
     * Let true === 1 class, false === 0 class
     */
    protected final Boolean[] outputs;
    private final int m;
    private final int n;

    /**
     * For Output array:
     * Let true === 1 class, false === 0 class
     */
    public LogisticRegression(Double[][] inputs, Boolean[] outputs) {
//        this.m = this.validateDatas(inputs, outputs);
        this.m = inputs.length;

        this.inputs = inputs;
        this.outputs = outputs;
        this.n = inputs[0].length;
    }

    public double costFunction(
            InsideFuncForSigmoidFunction insideFuncForSigmoidFunction,
            Double[] weights,
            Double bias
    ) {
//        this.areWeightsCountEqualN(scaledWeights);

        double cost = 0.;
        for (int i = 0; i < this.m; i++){
            cost += this.lossFunction(i, insideFuncForSigmoidFunction, weights, bias);
        }
        return cost / ((double) this.m);
    }

    public double J(
            InsideFuncForSigmoidFunction insideFuncForSigmoidFunction,
            Double[] weights,
            Double bias
    ) {
        return this.costFunction(insideFuncForSigmoidFunction, weights, bias);
    }

    private double lossFunction(
            int iThData,
            InsideFuncForSigmoidFunction insideFuncForSigmoidFunction,
            Double[] weights,
            Double bias
    ) {
//        this.areWeightsCountEqualN(scaledWeights);

        double loss = 0.0;
        if (this.outputs[iThData])
            loss = - Math.log(
                    this.sigmoidFunction(insideFuncForSigmoidFunction, this.inputs[iThData], weights, bias)
            );
        else
            loss = - Math.log(
                    1. - this.sigmoidFunction(insideFuncForSigmoidFunction, this.inputs[iThData], weights, bias)
            );

        return loss;
    }

    public double sigmoidFunction(
            InsideFuncForSigmoidFunction insideFuncForSigmoidFunction,
            Double[] iThInputValues,
            Double[] weights,
            Double bias
    ){
//        this.areWeightsCountEqualN(scaledWeights);

        return 1. / (1. + Math.pow(
                Math.E, -insideFuncForSigmoidFunction.insideFuncForSigmoidFunc(iThInputValues, weights, bias)
        ));
    }

    private int validateDatas(double[][] inputs, boolean[] outputs) {
        final int m = inputs.length;
        if (m != outputs.length)
            throw new InputAndOutputCountMismatchException(m, outputs.length);

        long n = inputs[0].length;
        for (double[] iThInputDatas : inputs)
            if (iThInputDatas.length != n)
                throw new EachInputDatasNotSameSizeException();

        return m;
    }

    protected void areWeightsCountEqualN(double[] weights){
        if (this.n != weights.length)
            throw new WeightArraySizeandFeatureCountMismatchException(this.n, weights.length);
    }

    public int getN() {
        return this.n;
    }

    public int getM() {
        return this.m;
    }
}