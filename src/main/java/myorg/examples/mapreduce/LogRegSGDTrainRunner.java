package myorg.examples.mapreduce;

import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import myorg.io.WeightVector;

public class LogRegSGDTrainRunner {
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser parser = new GenericOptionsParser(conf, args);
        args = parser.getRemainingArgs();

        if (args.length < 2) {
            System.err.println("Usage: input output");
            return;
        }
        String input = args[0];
        String output = args[1];
        String weightPath = args.length > 2 ? args[2] : "";

        int dim = 1 << 24;
        float eta0 = 1e-1f;
        float lambda = 1e-6f;

        conf.setInt(LogRegSGDTrainMapper.DIMENSION_CONFNAME, dim);
        conf.setFloat(LogRegSGDTrainMapper.ETA0_CONFNAME, eta0);
        conf.setFloat(LogRegSGDTrainMapper.LAMBDA_CONFNAME, lambda);
        conf.setInt(WeightVectorAverageReducer.DIMENSION_CONFNAME, dim);

        if (weightPath != "") {
            String cacheName = "weight";
            conf.set(LogRegSGDTestMapper.WEIGHTFILE_CONFNAME, cacheName);
            DistributedCache.createSymlink(conf);
            DistributedCache.addCacheFile(new URI(weightPath + "#" + cacheName), conf);
        }

        Job job = new Job(conf, "logistic regression SGD train");
        job.setJarByClass(LogRegSGDTrainRunner.class);
        job.setMapperClass(LogRegSGDTrainMapper.class);
        job.setReducerClass(WeightVectorAverageReducer.class);

        job.setNumReduceTasks(1);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(WeightVector.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(WeightVector.class);

        FileInputFormat.addInputPath(job, new Path(input));

        FileOutputFormat.setOutputPath(job, new Path(output));
        FileOutputFormat.setCompressOutput(job, true);
        FileOutputFormat.setOutputCompressorClass(job, org.apache.hadoop.io.compress.GzipCodec.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        job.waitForCompletion(true);
    }
}

