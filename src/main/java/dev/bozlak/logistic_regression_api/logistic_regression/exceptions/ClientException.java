package dev.bozlak.logistic_regression_api.logistic_regression.exceptions;

public class ClientException extends RuntimeException {
    public ClientException(String message) {
        super(message);
    }
}
