package myorg.examples.mapreduce;

import java.io.File;
import java.io.IOException;
import java.io.EOFException;
import java.util.Random;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.util.SVMLightFormatParser;

public class LinearLearnerTestMapper extends Mapper<Object, Text, FloatWritable, IntWritable> {

    public static String WEIGHTFILE_CONFNAME = "myorg.examples.hadoop.LinearLearnerTestMapper.weightFile";
    public static String LOGISTIC_CONFNAME = "myorg.examples.hadoop.LinearLearnerTestMapper.logistic";

    private String weightFile;
    private WeightVector weight;
    private FeatureVector datum;
    private boolean isBiasTermUsed = true;
    private boolean isLogisticUsed = false;

    private FloatWritable outKey;
    private IntWritable outVal;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

        weightFile = context.getConfiguration().get(WEIGHTFILE_CONFNAME, "weight");
        weight = new WeightVector();
        datum = new FeatureVector();
        isBiasTermUsed = true;
        isLogisticUsed = context.getConfiguration().getBoolean(LOGISTIC_CONFNAME, false);

        outKey = new FloatWritable();
        outVal = new IntWritable();

        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.getLocal(conf);

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
                } finally {
                    reader.close();
                }
            }
        }

    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        SVMLightFormatParser.parse(value.toString(), datum, isBiasTermUsed);

        float prediction = weight.innerProduct(datum);
        if (isLogisticUsed) {
            prediction = 1.0f / (1.0f + (float)Math.exp(-prediction));
        }
        
        outKey.set(prediction);
        outVal.set((int)datum.getLabel());

        context.write(outKey, outVal);
    }
}

