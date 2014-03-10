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

import myorg.common.LinearLearner;
import myorg.common.EtaCalculator;
import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.io.PartedWeightVector;
import myorg.util.SVMLightFormatParser;
import myorg.classifier.LogRegSGDLearner;
import myorg.regression.PARegLearner;
import myorg.regression.AROWRegLearner;

public class LinearLearnerTrainMapper extends Mapper<Object, Text, IntWritable, PartedWeightVector> {

    public enum ModelType {
        LOG_REG_SGD, PA_REG, AROW_REG
    }

    public static String MODELTYPE_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.modelType";
    public static String DIMENSION_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.dim";
    public static String ETA0_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.eta0";
    public static String LAMBDA_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.lambda";
    public static String C_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.C";
    public static String EPSILON_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.epsilon";
    public static String R_CONFNAME = "myorg.examples.hadoop.LinearLearnerTrainMapper.r";

    private LinearLearner learner;
    private FeatureVector datum;

    private boolean isBiasTermUsed = true;
    private int i = 0;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();

        ModelType modelType = conf.getEnum(MODELTYPE_CONFNAME, ModelType.LOG_REG_SGD);
        int dim    = conf.getInt(DIMENSION_CONFNAME, 1 << 24);
        float eta0   = conf.getFloat(ETA0_CONFNAME, 1e-1f);
        float lambda = conf.getFloat(LAMBDA_CONFNAME, 1e-6f);
        float C   = conf.getFloat(C_CONFNAME, 1.0f);
        float epsilon = conf.getFloat(EPSILON_CONFNAME, 0.1f);
        float r = conf.getFloat(R_CONFNAME, 1.0f);

        datum = new FeatureVector();
        isBiasTermUsed = true;

        FileSystem fs = FileSystem.getLocal(conf);

        WeightVector weight = null;
        WeightVector sigma = null;

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

        if (modelType == ModelType.LOG_REG_SGD) {
            if (eta0 <= 0.0f) {
                throw new RuntimeException("eta0 is less than or equal to 0.0");
            } else if (lambda <= 0.0f) {
                throw new RuntimeException("lambda is less than or equal to 0");
            }
            System.err.println(String.format("Logistic Regression SGD, eta0 = %f, lambda = %f", eta0, lambda));
            learner = new LogRegSGDLearner(0, lambda, new EtaCalculator(eta0, lambda), weight);

        } else if (modelType == ModelType.PA_REG) {
            if (epsilon <= 0.0f) {
                throw new RuntimeException("epsilon is less than or equal to 0.0");
            } else if (C <= 0.0f) {
                throw new RuntimeException("C is less than or equal to 0");
            }
            System.err.println(String.format("PA Regression, C = %f, epsilon = %f", C, epsilon));
            learner = new PARegLearner(PARegLearner.PAType.PA1, C, epsilon, weight);

        } else if (modelType == ModelType.AROW_REG) {
            if (r <= 0.0f) {
                throw new RuntimeException("r is less than or equal to 0.0");
            }

            sigma = new WeightVector(weight.getDimensions());
            for (int i = 0; i < sigma.getDimensions(); i++) {
                sigma.setValue(i, 1.0f);
            }

            System.err.println(String.format("AROW Regression, r = %f", r));
            learner = new AROWRegLearner(r, weight, sigma);

        }

    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        SVMLightFormatParser.parse(value.toString(), datum, isBiasTermUsed);
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

