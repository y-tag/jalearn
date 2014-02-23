package myorg.examples.mapreduce;

import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparator;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import myorg.io.WeightVector;

public class LogRegSGDTestRunner {

    public static class ReverseFloatWritableComparator implements RawComparator<FloatWritable> {
        private static final WritableComparator floatWritableComparator = new FloatWritable.Comparator();

        @Override
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return floatWritableComparator.compare(b2, s2, l2, b1, s1, l1);
        }

        @Override
        public int compare(FloatWritable o1, FloatWritable o2) {
            float diff = o2.get() - o1.get();
            return (diff == 0.0) ? 0 : ((diff > 0) ? +1 : -1);
        }

    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser parser = new GenericOptionsParser(conf, args);
        args = parser.getRemainingArgs();

        if (args.length < 3) {
            System.err.println("Usage: input output weight");
            return;
        }
        String input = args[0];
        String output = args[1];
        String weightPath = args[2];

        String cacheName = "weight";

        conf.set(LogRegSGDTestMapper.WEIGHTFILE_CONFNAME, cacheName);

        DistributedCache.createSymlink(conf);
        DistributedCache.addCacheFile(new URI(weightPath + "#" + cacheName), conf);

        Job job = new Job(conf, "logistic regression SGD train");
        job.setJarByClass(LogRegSGDTestRunner.class);
        job.setMapperClass(LogRegSGDTestMapper.class);
        job.setReducerClass(LogRegSGDTestReducer.class);

        job.setSortComparatorClass(ReverseFloatWritableComparator.class);

        job.setNumReduceTasks(1);

        job.setMapOutputKeyClass(FloatWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(FloatWritable.class);

        FileInputFormat.addInputPath(job, new Path(input));

        FileOutputFormat.setOutputPath(job, new Path(output));
        FileOutputFormat.setCompressOutput(job, true);
        FileOutputFormat.setOutputCompressorClass(job, org.apache.hadoop.io.compress.GzipCodec.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.waitForCompletion(true);
    }
}

