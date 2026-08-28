package dev.bozlak.logistic_regression_api.logistic_regression.exceptions;

public class InputAndOutputCountMismatchException extends ClientException {
    public InputAndOutputCountMismatchException(long inputsCount, long outputCounts) {
        super(
                "Input size (" + inputsCount + ") and output size (" + outputCounts + ") not same !!"
        );
    }
}