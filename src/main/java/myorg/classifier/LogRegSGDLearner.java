package myorg.classifier;

import java.util.List;
import java.util.Random;

import myorg.common.LinearLearner;
import myorg.common.EtaCalculator;
import myorg.io.FeatureVector;
import myorg.io.WeightVector;

public class LogRegSGDLearner implements LinearLearner {

    private long n;
    private float lambda;
    private EtaCalculator eCalc;
    private WeightVector w;

    public LogRegSGDLearner(long n, float lambda, EtaCalculator eCalc, WeightVector w) {
        this.n = n;
        this.lambda = lambda;
        this.eCalc = eCalc;
        this.w  = w;
    }

    @Override
    public float learn(FeatureVector fVec) {
        return learnWithStochasticOneStep(fVec, eCalc.get(n++), lambda, w);
    }

    @Override
    public void setWeight(WeightVector w) {
        this.w = w;
    }

    @Override
    public WeightVector getWeight() {
        return w;
    }

    public static float learnWithStochasticOneStep(
       FeatureVector fVec, float eta, float lambda, WeightVector w
    ) {
        float y = (fVec.getLabel() > 0.0f) ? 1.0f : -1.0f;
        float ip = w.innerProduct(fVec);

        w.scale(1.0f - eta * lambda);

        float z = y * ip;
        float coef = 0.0f;
        if (z < 10.0f) {
            coef = (float)(1.0 / (1.0 + Math.exp(z)));
        } else {
            double ez = Math.exp(-z);
            coef = (float)(ez / (ez + 1.0));
        }

        w.addVector(fVec, y * coef * eta);

        return z;
    }
}

