package myorg.common;

public class EtaCalculator {

    private float eta0;
    private float lambda;

    public EtaCalculator(float eta0, float lambda) {
        this.eta0 = eta0;
        this.lambda = lambda;
    }

    public float get(long i) {
        return eta0 / (1.0f + eta0 * lambda * i);
    }
}

