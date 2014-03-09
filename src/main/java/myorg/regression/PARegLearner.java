package myorg.regression;

import java.util.List;
import java.util.Random;

import myorg.common.LinearLearner;
import myorg.io.FeatureVector;
import myorg.io.WeightVector;

public class PARegLearner implements LinearLearner {

    public enum PAType {
        PA, PA1, PA2
    }

    // See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    public static class VarianceCalculator {
        long n;
        float mean;
        float M2;

        public VarianceCalculator() {
            long n = 0;
            float mean = 0.0f;
            float M2 = 0.0f;
        }

        public void add(float x) {
            n++;
            float delta = x - mean;
            mean += delta / n;
            M2 += delta * (x - mean);
        }

        public float get() {
            return (n < 2) ? 0.0f : (M2 / (n - 1));
        }
    }

    private PAType paType;
    private float C;
    private float epsilon;
    private VarianceCalculator vc;
    private WeightVector w;

    public PARegLearner(PAType paType, float C, float epsilon, WeightVector w) {
        this.paType = paType;
        this.C = C;
        this.epsilon = epsilon;
        this.vc = new VarianceCalculator();
        this.w  = w;
    }

    @Override
    public float learn(FeatureVector fVec) {
        vc.add(fVec.getLabel());
        float stddev = (float)Math.sqrt(vc.get());
        return learnWithOneStep(fVec, paType, C, epsilon, stddev, w);
    }

    @Override
    public void setWeight(WeightVector w) {
        this.w = w;
    }

    @Override
    public WeightVector getWeight() {
        return w;
    }

    public static float learnWithOneStep(
       FeatureVector fVec, PAType paType, float C, float epsilon, float stddev, WeightVector w
    ) {
        float ip = w.innerProduct(fVec);
        float diff = fVec.getLabel() - ip;
        float loss = Math.abs(diff) - epsilon * stddev;

        if (loss <= 0.0f) { return 0.0f; }

        int sign = diff > 0.0f ? +1 : -1;
        float coef = calcEta(paType, loss, fVec.getSquaredNorm(), C);

        w.addVector(fVec, sign * coef);

        return loss;
    }

    public static float calcEta(PAType paType, float loss, float squaredNorm, float C) {
        if (paType == PAType.PA1) {
            return Math.min(C, loss / squaredNorm);
        } else if (paType == PAType.PA2) {
            return loss / (squaredNorm + (0.5f / C));
        } else {
            return loss / squaredNorm;
        }
    }
}

