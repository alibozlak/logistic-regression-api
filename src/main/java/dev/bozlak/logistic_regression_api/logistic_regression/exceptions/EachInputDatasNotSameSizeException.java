package dev.bozlak.logistic_regression_api.logistic_regression.exceptions;

public class EachInputDatasNotSameSizeException extends ClientException {
    public EachInputDatasNotSameSizeException() {
        super("Each input data array not same size !!");
    }
}
