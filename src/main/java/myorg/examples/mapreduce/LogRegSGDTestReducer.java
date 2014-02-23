package myorg.examples.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Reducer;

public class LogRegSGDTestReducer extends Reducer<FloatWritable, IntWritable, IntWritable, FloatWritable> {

    double curAUC;
    double oldPosSum;
    double posSum;
    double negSum;
    double negNum;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        curAUC = 0.0;
        oldPosSum = 0.0;
        posSum = 0.0;
        negSum = 0.0;
        negNum = 0.0;
    }

    @Override
    protected void reduce(FloatWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        for (IntWritable value : values) {
            //context.write(key, value);
            if (value.get() > 0) {
                posSum += 1.0;
            } else {
                negNum += 1.0;
                negSum += 1.0;
            }
        }

        curAUC += (oldPosSum + posSum) * negNum / 2.0;
        oldPosSum = posSum;
        negNum = 0.0;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        curAUC += (oldPosSum + posSum) * negNum / 2.0;
        curAUC /= (posSum * negSum);
        context.write(new IntWritable((int)(posSum + negSum)), new FloatWritable((float)curAUC));
    }

}
