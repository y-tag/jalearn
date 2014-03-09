package myorg.regression;

import java.util.List;
import java.util.Random;

import myorg.common.LinearLearner;
import myorg.io.FeatureVector;
import myorg.io.WeightVector;

public class AROWRegLearner implements LinearLearner {

    private float r;
    private WeightVector w;
    private WeightVector sigma;

    public AROWRegLearner(float r, WeightVector w, WeightVector sigma) {
        this.r = r;
        this.w  = w;
        this.sigma = sigma;
    }

    @Override
    public float learn(FeatureVector fVec) {
        return learnWithOneStep(fVec, r, w, sigma);
    }

    @Override
    public void setWeight(WeightVector w) {
        this.w = w;
    }

    @Override
    public WeightVector getWeight() {
        return w;
    }

    public void setSigma(WeightVector sigma) {
        this.sigma = sigma;
    }

    public WeightVector getSigma() {
        return sigma;
    }

    public static float learnWithOneStep(
       FeatureVector fVec, float r, WeightVector w, WeightVector sigma
    ) {
        float ip = w.innerProduct(fVec);

        float confidence = 0.0f;
        for (int i = 0; i < fVec.getNonZeroNum(); i++) {
            int   idx = fVec.getIndexAt(i);
            float val = fVec.getValueAt(i);
            confidence += val * val * sigma.getValue(idx);
        }

        float diff = fVec.getLabel() - ip;
        float beta  = 1.0f / (r + confidence);
        float alpha = diff * beta;

        for (int i = 0; i < fVec.getNonZeroNum(); i++) {
            int   idx = fVec.getIndexAt(i);
            float val = fVec.getValueAt(i);

            float sx = sigma.getValue(idx) * val;
            w.setValue(idx, w.getValue(idx) + alpha * sx);
            sigma.setValue(idx, sigma.getValue(idx) - beta * sx * sx);
        }

        float loss = diff * diff;
        return loss;
    }
}

