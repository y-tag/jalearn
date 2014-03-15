package myorg.examples.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import myorg.io.WeightVector;
import myorg.io.PartedWeightVector;

public class WeightVectorAverageReducer extends Reducer<VectorInfo, PartedWeightVector, Text, WeightVector> {
    private int weightNum;
    private int sigmaNum;
    private WeightVector weight;
    private WeightVector sigma;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        weightNum = 0;
        sigmaNum = 0;
        weight = new WeightVector();
        sigma = new WeightVector();
    }

    @Override
    protected void reduce(VectorInfo key, Iterable<PartedWeightVector> values, Context context) throws IOException, InterruptedException {
        String vectorType = key.getVectorType();

        if (vectorType.equals("weight")) {
            for (PartedWeightVector value : values) {
                weight.addVector(value);
            }
            weightNum++;

        } else if (vectorType.equals("sigma")) {
            for (PartedWeightVector value : values) {
                sigma.addVector(value);
            }
            sigmaNum++;
        }

    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (weightNum > 0) {
            weight.scale(1.0f / weightNum);
            context.write(new Text("weight"), weight);
        }
        if (sigmaNum > 0) {
            sigma.scale(1.0f / sigmaNum);
            context.write(new Text("sigma"), sigma);
        }
    }
}

