package myorg.examples.mapreduce;

import java.io.IOException;
import java.io.EOFException;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
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
import myorg.io.PartedWeightVector;
import myorg.util.SVMLightFormatParser;
import myorg.regression.PARegLearner;

public class PARegTrainMapper extends Mapper<Object, Text, IntWritable, PartedWeightVector> {

    public static String DIMENSION_CONFNAME = "myorg.examples.hadoop.PARegTrainMapper.dim";
    public static String WEIGHTFILE_CONFNAME = "myorg.examples.hadoop.PARegTrainMapper.weightFile";
    public static String C_CONFNAME = "myorg.examples.hadoop.PARegTrainMapper.C";
    public static String EPSILON_CONFNAME = "myorg.examples.hadoop.PARegTrainMapper.epsilon";

    private PARegLearner learner;
    private FeatureVector datum;

    private boolean isBiasTermUsed = true;
    private int i = 0;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

        int dim    = context.getConfiguration().getInt(DIMENSION_CONFNAME, 1 << 24);
        float C   = context.getConfiguration().getFloat(C_CONFNAME, 1.0f);
        float epsilon = context.getConfiguration().getFloat(EPSILON_CONFNAME, 0.1f);

        datum = new FeatureVector();
        isBiasTermUsed = true;

        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.getLocal(conf);

        WeightVector weight = null;
        
        Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);

        if (cacheFiles != null) {
            for (int i = 0; i < cacheFiles.length; i++) {
                System.err.println(String.format("read cacheFiles[%d]: %s", i, cacheFiles[i].toString()));
                SequenceFile.Reader reader = new SequenceFile.Reader(fs, cacheFiles[i], conf);
                try {
                    Class<?> keyClass = reader.getKeyClass();
                    Class<?> valClass = reader.getValueClass();

                    if (keyClass == Text.class && valClass == WeightVector.class) {
                        Text key = new Text();
                        weight = new WeightVector();

                        while (reader.next(key, weight)) {
                            break;
                        }
                    }

                } catch (Exception e) {
                    weight = null;
                } finally {
                    reader.close();
                }
            }
        }
        
        if (weight == null) {
            System.err.println("new weight vector is created");
            if (dim <= 0) {
                throw new RuntimeException("dim is less than or equal to 0");
            }
            weight = new WeightVector(dim);
        }

        if (epsilon <= 0.0f) {
            throw new RuntimeException("epsilon is less than or equal to 0.0");
        } else if (C <= 0.0f) {
            throw new RuntimeException("C is less than or equal to 0");
        }

        learner = new PARegLearner(PARegLearner.PAType.PA1, C, epsilon, weight);

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

        int dim = learner.getWeight().getDimensions();
        int splitSize = 1 << 20;
        if (splitSize > dim) { splitSize = dim; }
        List<PartedWeightVector> list = learner.getWeight().splitAsPartedWeightVector(splitSize);

        for (PartedWeightVector pwv : list) {
            context.write(new IntWritable(id), pwv);
        }
    }
}

