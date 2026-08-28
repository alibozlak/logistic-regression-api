package dev.bozlak.logistic_regression_api.logistic_regression.use_linear_func;

import dev.bozlak.logistic_regression_api.logistic_regression.dtos.responses.SuccessResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/linear-regression/use-linear-func")
public class ApiEndpoint {

    private final ApiService apiService;

    public ApiEndpoint(ApiService apiService) {
        this.apiService = apiService;
    }

//    @PostMapping
//    public ResponseEntity<SuccessResponse> trainModel(
//            @RequestBody dev.bozlak.logistic_regression_api.logistic_regression.dtos.requests.RequestBody requestBody
//    ) {
//    }

}
