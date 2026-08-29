package dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests;

import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.ClientException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.EachInputDatasNotSameSizeException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.InputAndOutputCountMismatchException;

public class RequestBodyForTrain {

    public final Double[][] inputs;
    public final Boolean[] outputs;
    public final Double learningRate;
    public final Integer loopCount;

    public RequestBodyForTrain(Double[][] inputs, Boolean[] outputs, Double learningRate, Integer loopCount) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.learningRate = learningRate;
        this.loopCount = loopCount;
    }

    public static void validateRequestBody(
            Double[][] inputs, Boolean[] outputs, Double learningRate, Integer loopCount
    ) {
        if (inputs.length == 0)
            throw new ClientException("Input array must not be an empty!!");

        if (learningRate == null || learningRate <= 0 || learningRate > 1)
            throw new ClientException("learningRate should be between (0,1] !! Yours = " + learningRate);

        if (loopCount == null || loopCount <= 0)
            throw new ClientException("loopCount must be greater than 0 !! Yours = " + loopCount);

        final int m = inputs.length;
        if (m != outputs.length)
            throw new InputAndOutputCountMismatchException(m, outputs.length);

        final int n = inputs[0].length;
        for (int i = 1; i < m; i++)
            if (n != inputs[i].length)
                throw new EachInputDatasNotSameSizeException();
    }
}
