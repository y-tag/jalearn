package myorg.examples.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import myorg.io.WeightVector;
import myorg.io.PartedWeightVector;

public class WeightVectorAverageReducer extends Reducer<IntWritable, PartedWeightVector, Text, WeightVector> {
    private int num;
    private WeightVector weight;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        num = 0;
        weight = new WeightVector();
    }

    @Override
    protected void reduce(IntWritable key, Iterable<PartedWeightVector> values, Context context) throws IOException, InterruptedException {
        for (PartedWeightVector value : values) {
            weight.addVector(value);
        }
        num++;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        weight.scale(1.0f / num);
        context.write(new Text(), weight);
    }
}
