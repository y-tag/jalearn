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
import myorg.io.PartedWeightVector;

public class LinearLearnerTrainRunner {

    public Job initLogRegSGD(Configuration conf, int iter, int nIter,
                             int dim, float eta0, float lambda, String weightPath) throws Exception {
        conf.setInt(LogRegSGDTrainMapper.DIMENSION_CONFNAME, dim);
        conf.setFloat(LogRegSGDTrainMapper.ETA0_CONFNAME, eta0);
        conf.setFloat(LogRegSGDTrainMapper.LAMBDA_CONFNAME, lambda);

        if (weightPath != null && weightPath != "") {
            String cacheName = "weight";
            DistributedCache.createSymlink(conf);
            DistributedCache.addCacheFile(new URI(weightPath + "#" + cacheName), conf);
        }

        String jobName = String.format("Logistic Regression SGD Train (%d in %d)", iter, nIter);

        Job job = new Job(conf, jobName);
        job.setJarByClass(LinearLearnerTrainRunner.class);
        job.setMapperClass(LogRegSGDTrainMapper.class);
        job.setReducerClass(WeightVectorAverageReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(PartedWeightVector.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(WeightVector.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        return job;
    }

    public Job initPAReg(Configuration conf, int iter, int nIter,
                         int dim, float C, float epsilon, String weightPath) throws Exception {
        conf.setInt(PARegTrainMapper.DIMENSION_CONFNAME, dim);
        conf.setFloat(PARegTrainMapper.C_CONFNAME, C);
        conf.setFloat(PARegTrainMapper.EPSILON_CONFNAME, epsilon);

        if (weightPath != null && weightPath != "") {
            String cacheName = "weight";
            DistributedCache.createSymlink(conf);
            DistributedCache.addCacheFile(new URI(weightPath + "#" + cacheName), conf);
        }

        String jobName = String.format("PassiveAggressive Regression Train (%d in %d)", iter, nIter);

        Job job = new Job(conf, jobName);
        job.setJarByClass(LinearLearnerTrainRunner.class);
        job.setMapperClass(PARegTrainMapper.class);
        job.setReducerClass(WeightVectorAverageReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(PartedWeightVector.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(WeightVector.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        return job;
    }

    public Job initAROWReg(Configuration conf, int iter, int nIter,
                           int dim, float r, String weightPath) throws Exception {
        conf.setInt(AROWRegTrainMapper.DIMENSION_CONFNAME, dim);
        conf.setFloat(AROWRegTrainMapper.R_CONFNAME, r);

        if (weightPath != null && weightPath != "") {
            String cacheName = "weight";
            DistributedCache.createSymlink(conf);
            DistributedCache.addCacheFile(new URI(weightPath + "#" + cacheName), conf);
        }

        String jobName = String.format("AROW Regression Train (%d in %d)", iter, nIter);

        Job job = new Job(conf, jobName);
        job.setJarByClass(LinearLearnerTrainRunner.class);
        job.setMapperClass(AROWRegTrainMapper.class);
        job.setReducerClass(WeightVectorAverageReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(PartedWeightVector.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(WeightVector.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        return job;
    }

    private static void printUsage(Options opts) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setWidth(80);
        formatter.printHelp("myorg.examples.mapreduce.LinearLearnerTrainRunner input output_base", opts);
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser parser = new GenericOptionsParser(conf, args);
        args = parser.getRemainingArgs();

        Options opts = new Options();
        opts.addOption("modelType", true,
            "Type of model and learning method.\n" +
            "0: Logistic regression using SGD and simple weight averaging\n" +
            "1: PassiveAggressive regression using simple weight averaging"
        );
        opts.addOption("dim", true, "Number of weight vector dimension.");
        opts.addOption("eta0", true, "Initial value of learning rate.");
        opts.addOption("lambda", true, "Regularization parameter.");
        opts.addOption("C", true, "Aggressiveness parameter.");
        opts.addOption("epsilon", true, "Regression tolerance.");
        opts.addOption("r", true, "Aggressiveness parameter.");
        opts.addOption("weightPath", true, "Initial weight file path.");
        opts.addOption("nIter", true, "Number of MapReduce Iterations.");

        CommandLine cliParser = new GnuParser().parse(opts, args);
        args = cliParser.getArgs();

        int modelType = Integer.parseInt(cliParser.getOptionValue("modelType", "0"));
        int dim = Integer.parseInt(cliParser.getOptionValue("dim", "16777216"));
        float eta0 = Float.parseFloat(cliParser.getOptionValue("eta0", "1e-1"));
        float lambda = Float.parseFloat(cliParser.getOptionValue("lambda", "1e-7"));
        float C = Float.parseFloat(cliParser.getOptionValue("C", "1.0"));
        float epsilon = Float.parseFloat(cliParser.getOptionValue("epsilon", "0.1"));
        float r = Float.parseFloat(cliParser.getOptionValue("r", "1.0"));
        String weightPath = cliParser.getOptionValue("weightPath", "");
        int nIter = Integer.parseInt(cliParser.getOptionValue("nIter", "1"));

        if (args.length != 2) {
            printUsage(opts);
            return;
        }
        String input = args[0];
        String outputBase = args[1];

        LinearLearnerTrainRunner trainRunner = new LinearLearnerTrainRunner();

        for (int iter = 0; iter < nIter; iter++) {
            String output = String.format("%s/iter-%05d", outputBase, iter);

            Job job = null;
            Configuration copiedConf = new Configuration(conf);

            if (modelType == 0) {
                job = trainRunner.initLogRegSGD(copiedConf, iter, nIter, dim, eta0, lambda, weightPath);
            } else if (modelType == 1) {
                job = trainRunner.initPAReg(copiedConf, iter, nIter, dim, C, epsilon, weightPath);
            } else if (modelType == 2) {
                job = trainRunner.initAROWReg(copiedConf, iter, nIter, dim, r, weightPath);
            } else {
                System.err.println("unkown modelType: " + modelType);
                return;
            }

            job.setNumReduceTasks(1);

            FileInputFormat.addInputPath(job, new Path(input));
            FileOutputFormat.setOutputPath(job, new Path(output));
            FileOutputFormat.setCompressOutput(job, true);
            FileOutputFormat.setOutputCompressorClass(job, org.apache.hadoop.io.compress.GzipCodec.class);

            job.waitForCompletion(true);

            weightPath = output + "/part-r-00000";
        }
    }
}

