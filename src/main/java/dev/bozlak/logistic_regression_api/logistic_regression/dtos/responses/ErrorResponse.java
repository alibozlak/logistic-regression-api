package dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses;

public class ErrorResponse {

    public final String errorMessage;

    public ErrorResponse(String errorMessage) {
        this.errorMessage = errorMessage;
    }

}
