package dev.bozlak.logistic_regression_api.logistic_regression.feature_column_scaling;

import org.springframework.stereotype.Service;

@Service
public class ScalingServiceImpl implements ScalingService {

    @Override
    public ScaledResult scale(Double[][] requestInputs) {
        final int n = requestInputs[0].length;
        final int m = requestInputs.length;
        Double[][] scaledInputs = new Double[m][n];
        Integer[] ratiosPower10 = new Integer[n];

        for (int j = 0; j < n; j++) {
            int digitsCount = ((int) Math.abs(requestInputs[0][j]) + "").length();
            ratiosPower10[j] = digitsCount - 1;
        }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                scaledInputs[i][j] = requestInputs[i][j] * Math.pow(10., - ratiosPower10[j]);

        return new ScaledResult(scaledInputs, ratiosPower10);
    }

}
