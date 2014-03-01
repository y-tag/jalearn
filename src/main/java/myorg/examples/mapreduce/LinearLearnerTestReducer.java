package myorg.examples.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Reducer;

public class LinearLearnerTestReducer extends Reducer<FloatWritable, IntWritable, Text, Text> {

    double curAUROC;
    double curAUPRC;
    double curRMSE;
    double curMAE;
    double curNLL;

    double oldPosSum;
    double oldPrecision;
    double posNum;
    double posSum;
    double negNum;
    double negSum;


    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        curAUROC = 0.0;
        curAUPRC = 0.0;
        curRMSE = 0.0;
        curMAE = 0.0;
        curNLL = 0.0;

        oldPosSum = 0.0;
        oldPrecision = 1.0;
        posNum = 0.0;
        posSum = 0.0;
        negNum = 0.0;
        negSum = 0.0;
    }

    @Override
    protected void reduce(FloatWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        for (IntWritable value : values) {
            if (value.get() > 0) {
                posNum += 1.0;
                posSum += 1.0;
            } else {
                negNum += 1.0;
                negSum += 1.0;
            }

            double y = (value.get() > 0) ? 1.0 : 0.0;
            curRMSE += Math.pow(y - key.get(), 2.0);
            curMAE += Math.abs(y - key.get());

            double p = Math.max(Math.min(key.get(), 1.0 - 1.0e-15), 1.0e-15);
            double tmp = (value.get() > 0) ? p : (1.0 - p);
            curNLL -= Math.log(tmp);
        }

        curAUROC += (oldPosSum + posSum) * negNum / 2.0;
        oldPosSum = posSum;
        negNum = 0.0;

        double precision = posSum / (posSum + negSum);
        curAUPRC += (oldPrecision + precision) * posNum / 2.0;
        oldPrecision = precision;
        posNum = 0.0;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        context.write(new Text("Number of samples"),
                      new Text(String.valueOf((long)(posSum + negSum))));

        curAUROC /= (posSum * negSum);
        context.write(new Text("AUC"),
                      new Text(String.format("%.7f", curAUROC)));

        curAUPRC /= posSum;
        context.write(new Text("AUPRC"),
                      new Text(String.format("%.7f", curAUPRC)));

        curRMSE = Math.sqrt(curRMSE / (posSum + negSum));
        context.write(new Text("RMSE"),
                      new Text(String.format("%.7f", curRMSE)));

        curMAE = curMAE / (posSum + negSum);
        context.write(new Text("MAE"),
                      new Text(String.format("%.7f", curMAE)));

        context.write(new Text("NLL"),
                      new Text(String.format("%.7f", curNLL)));
    }

}
