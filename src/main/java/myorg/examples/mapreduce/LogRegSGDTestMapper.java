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
import myorg.classifier.LogRegSGDLearner;

public class LogRegSGDTestMapper extends Mapper<Object, Text, FloatWritable, IntWritable> {

    public static String WEIGHTFILE_CONFNAME = "myorg.examples.hadoop.LogRegSGDTestMapper.weightFile";

    private String weightFile;
    private WeightVector weight;
    private FeatureVector datum;
    private boolean isBiasTermUsed = true;

    private FloatWritable outKey;
    private IntWritable outVal;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {

        weightFile = context.getConfiguration().get(WEIGHTFILE_CONFNAME, "weight");
        weight = new WeightVector();
        datum = new FeatureVector();
        isBiasTermUsed = true;

        outKey = new FloatWritable();
        outVal = new IntWritable();

        Configuration conf = context.getConfiguration();
        Path path = new Path(weightFile);
        FileSystem fs = FileSystem.getLocal(conf);

        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
        try {
            Class<?> keyClass = reader.getKeyClass();

            Writable key;
            if (keyClass == NullWritable.class) {
                key = NullWritable.get();
            } else {
                key = (Writable) keyClass.newInstance();
            }

            while (reader.next(key, weight)) {
                break;
            }
        } catch (Exception e) {
        } finally {
            reader.close();
        }
    

    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        SVMLightFormatParser.parse(value.toString(), datum, isBiasTermUsed);

        float prediction = weight.innerProduct(datum);
        prediction = 1.0f / (1.0f + (float)Math.exp(-prediction));
        
        outKey.set(prediction);
        outVal.set((int)datum.getLabel());

        context.write(outKey, outVal);
    }
}

