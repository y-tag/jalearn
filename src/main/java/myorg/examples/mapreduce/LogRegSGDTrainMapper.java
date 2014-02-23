package myorg.examples.mapreduce;

import java.io.IOException;
import java.io.EOFException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.util.SVMLightFormatParser;
import myorg.classifier.LogRegSGDLearner;

public class LogRegSGDTrainMapper extends Mapper<Object, Text, IntWritable, WeightVector> {

    public static String DIMENSION_CONFNAME = "myorg.examples.hadoop.LogRegSGDTrainMapper.dim";
    public static String WEIGHTFILE_CONFNAME = "myorg.examples.hadoop.LogRegSGDTrainMapper.weightFile";
    public static String ETA0_CONFNAME = "myorg.examples.hadoop.LogRegSGDTrainMapper.eta0";
    public static String LAMBDA_CONFNAME = "myorg.examples.hadoop.LogRegSGDTrainMapper.lambda";

    private float eta0;
    private float lambda;
    private WeightVector weight = new WeightVector();
    private FeatureVector datum = new FeatureVector();

    private boolean isBiasTermUsed = true;
    private int i = 0;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

        int dim    = context.getConfiguration().getInt(DIMENSION_CONFNAME, 1 << 24);
        String weightFile = context.getConfiguration().get(WEIGHTFILE_CONFNAME, "weight");
        eta0   = context.getConfiguration().getFloat(ETA0_CONFNAME, 1e-1f);
        lambda = context.getConfiguration().getFloat(LAMBDA_CONFNAME, 1e-6f);

        isBiasTermUsed = true;
        i = 0;

        Configuration conf = context.getConfiguration();
        Path weightPath = new Path(weightFile);
        FileSystem fs = FileSystem.getLocal(conf);

        weight = null;
        
        if (fs.exists(weightPath)) {
            System.err.println("file exists: " + weightPath.toString());
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, weightPath, conf);
            try {
                Class<?> keyClass = reader.getKeyClass();

                Writable key;
                if (keyClass == NullWritable.class) {
                    key = NullWritable.get();
                } else {
                    key = (Writable) keyClass.newInstance();
                }

                weight = new WeightVector();

                while (reader.next(key, weight)) {
                    break;
                }
            } catch (Exception e) {
                weight = null;
            } finally {
                reader.close();
            }
        }
        
        if (weight == null) {
            System.err.println("file does not exist: " + weightPath.toString());
            if (dim <= 0) {
                throw new RuntimeException("dim is less than or equal to 0");
            }
            weight = new WeightVector(dim);
        }

        if (eta0 <= 0.0f) {
            throw new RuntimeException("eta0 is less than or equal to 0.0");
        } else if (lambda <= 0.0f) {
            throw new RuntimeException("lambda is less than or equal to 0");
        }

    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        SVMLightFormatParser.parse(value.toString(), datum, isBiasTermUsed);

        float eta = eta0 / (1.0f + eta0 * lambda * i);
        LogRegSGDLearner.learnWithStochasticOneStep(datum, eta, lambda, weight);
        i++;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        int id = context.getTaskAttemptID().getTaskID().getId();
        context.write(new IntWritable(id), weight);
    }
}

