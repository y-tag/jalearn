package myorg.examples.mapreduce;

import java.net.URI;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

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

public class LinearClassifierTrainRunner {

    public static void initOptions(Options opts) {
        opts.addOption("modelType", true, "Type of model and learning method.");
        opts.addOption("dim", true, "Number of weight vector dimension.");
        opts.addOption("eta0", true, "Initial value of learning rate.");
        opts.addOption("lambda", true, "Regularization parameter.");
        opts.addOption("weightPath", true, "Initial weight file path.");
    }

    private static void printUsage(Options opts) {
        new HelpFormatter().printHelp("myorg.examples.mapreduce.LinearClassifierTrainRunner input output", opts);
    }

    public Job initLogRegSGD(Configuration conf, String input, String output,
                             int dim, float eta0, float lambda, String weightPath) throws Exception {
        conf.setInt(LogRegSGDTrainMapper.DIMENSION_CONFNAME, dim);
        conf.setFloat(LogRegSGDTrainMapper.ETA0_CONFNAME, eta0);
        conf.setFloat(LogRegSGDTrainMapper.LAMBDA_CONFNAME, lambda);
        conf.setInt(WeightVectorAverageReducer.DIMENSION_CONFNAME, dim);

        if (weightPath != null && weightPath != "") {
            String cacheName = "weight";
            conf.set(LogRegSGDTestMapper.WEIGHTFILE_CONFNAME, cacheName);
            DistributedCache.createSymlink(conf);
            DistributedCache.addCacheFile(new URI(weightPath + "#" + cacheName), conf);
        }

        Job job = new Job(conf, "Logistic Regression SGD Train");
        job.setJarByClass(LinearClassifierTrainRunner.class);
        job.setMapperClass(LogRegSGDTrainMapper.class);
        job.setReducerClass(WeightVectorAverageReducer.class);

        job.setNumReduceTasks(1);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(WeightVector.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(WeightVector.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        return job;
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser parser = new GenericOptionsParser(conf, args);
        args = parser.getRemainingArgs();

        Options opts = new Options();
        initOptions(opts);
        CommandLine cliParser = new GnuParser().parse(opts, args);
        args = cliParser.getArgs();

        if (args.length != 2) {
            printUsage(opts);
            return;
        }
        String input = args[0];
        String output = args[1];

        int modelType = Integer.parseInt(cliParser.getOptionValue("modelType", "0"));
        int dim = Integer.parseInt(cliParser.getOptionValue("dim", "16777216"));
        float eta0 = Float.parseFloat(cliParser.getOptionValue("eta0", "1e-1"));
        float lambda = Float.parseFloat(cliParser.getOptionValue("lambda", "1e-6"));
        String weightPath = cliParser.getOptionValue("weightPath", "");

        LinearClassifierTrainRunner trainRunner = new LinearClassifierTrainRunner();

        Job job = trainRunner.initLogRegSGD(conf, input, output, dim, eta0, lambda, weightPath);

        FileOutputFormat.setCompressOutput(job, true);
        FileOutputFormat.setOutputCompressorClass(job, org.apache.hadoop.io.compress.GzipCodec.class);

        job.waitForCompletion(true);
    }
}

