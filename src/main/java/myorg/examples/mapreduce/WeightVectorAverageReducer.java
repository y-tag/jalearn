package myorg.examples.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import myorg.io.WeightVector;

public class WeightVectorAverageReducer extends Reducer<IntWritable, WeightVector, Text, WeightVector> {

    public static String DIMENSION_CONFNAME = "myorg.examples.hadoop.WeightVectorAverageReducer.dim";

    private int dim;
    private int num;
    private WeightVector weight;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        dim = context.getConfiguration().getInt(DIMENSION_CONFNAME, 1 << 24);
        num = 0;

        weight = new WeightVector(dim);
    }

    @Override
    protected void reduce(IntWritable key, Iterable<WeightVector> values, Context context) throws IOException, InterruptedException {
        for (WeightVector value : values) {
            weight.addVector(value);
            num++;
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        weight.scale(1.0f / num);
        context.write(new Text(), weight);
    }
}
