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

import myorg.common.LinearLearner;
import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.util.SVMLightFormatParser;
import myorg.regression.AROWRegLearner;

public class AROWRegTrainMapper extends Mapper<Object, Text, IntWritable, WeightVector> {

    public static String DIMENSION_CONFNAME = "myorg.examples.hadoop.AROWRegTrainMapper.dim";
    public static String WEIGHTFILE_CONFNAME = "myorg.examples.hadoop.AROWRegTrainMapper.weightFile";
    public static String R_CONFNAME = "myorg.examples.hadoop.AROWRegTrainMapper.r";

    private LinearLearner learner;
    private FeatureVector datum;

    private boolean isBiasTermUsed = true;
    private int i = 0;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

        int dim    = context.getConfiguration().getInt(DIMENSION_CONFNAME, 1 << 24);
        String weightFile = context.getConfiguration().get(WEIGHTFILE_CONFNAME, "weight");
        float r = context.getConfiguration().getFloat(R_CONFNAME, 1.0f);

        datum = new FeatureVector();
        isBiasTermUsed = true;

        Configuration conf = context.getConfiguration();
        Path weightPath = new Path(weightFile);
        FileSystem fs = FileSystem.getLocal(conf);

        WeightVector weight = null;
        WeightVector sigma = null;
        
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

        if (r <= 0.0f) {
            throw new RuntimeException("r is less than or equal to 0.0");
        }

        sigma = new WeightVector(weight.getDimensions());
        for (int i = 0; i < sigma.getDimensions(); i++) {
            sigma.setValue(i, 1.0f);
        }

        learner = new AROWRegLearner(r, weight, sigma);

    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        SVMLightFormatParser.parse(value.toString(), datum, isBiasTermUsed);
        if (datum.getLabel() < 0.0f) { datum.setLabel(0.0f); }
        else                         { datum.setLabel(1.0f); }
        learner.learn(datum);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        int id = context.getTaskAttemptID().getTaskID().getId();
        context.write(new IntWritable(id), learner.getWeight());
    }
}

