package dev.bozlak.logistic_regression_api;

import dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses.ErrorResponse;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.ClientException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.EachInputDatasNotSameSizeException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.InputAndOutputCountMismatchException;
import dev.bozlak.logistic_regression_api.logistic_regression.exceptions.WeightArraySizeandFeatureCountMismatchException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.stream.Collectors;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidationExceptions(MethodArgumentNotValidException ex){
        String errorMessage = ex.getBindingResult().getFieldErrors().stream()
                .map(FieldError::getDefaultMessage).collect(Collectors.joining(", "));

        return new ResponseEntity<>(
                new ErrorResponse(errorMessage),
                HttpStatus.BAD_REQUEST
        );
    }

    @ExceptionHandler({
            EachInputDatasNotSameSizeException.class,
            InputAndOutputCountMismatchException.class,
            WeightArraySizeandFeatureCountMismatchException.class
    })
    public ResponseEntity<ErrorResponse> handleClientException(ClientException clientException) {

        return new ResponseEntity<>(
                new ErrorResponse(clientException.getMessage()),
                HttpStatus.BAD_REQUEST
        );
    }


    // ********************************************************************************
    // *********************** Other Exceptions Handle ********************************
    // ********************************************************************************

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleUnknownException(Exception exception) {
        return new ResponseEntity<>(
                new ErrorResponse(exception.getMessage()),
                HttpStatus.INTERNAL_SERVER_ERROR
        );
    }

}
