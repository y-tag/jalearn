package myorg.classifier;

import myorg.io.FeatureVector;
import myorg.io.WeightVector;

public interface LinearLearner {

    float learn(FeatureVector fVec);
    public void setWeight(WeightVector w);
    public WeightVector getWeight();

}

