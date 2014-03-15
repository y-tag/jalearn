package myorg.examples.local;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import myorg.common.LinearLearner;
import myorg.common.EtaCalculator;
import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.util.SVMLightFormatParser;
import myorg.classifier.LogRegSGDLearner;
import myorg.regression.PARegLearner;
import myorg.regression.AROWRegLearner;

public class LinearLearnerTrain {

    public static void main(String[] args) throws Exception {
        Options opts = new Options();
        opts.addOption("modelType", true,
            "Type of model and learning method.\n" +
            "0: Logistic regression using SGD\n" +
            "1: PassiveAggressive regression\n" +
            "2: AROW regression"
        );
        opts.addOption("dim", true, "Number of weight vector dimension.");
        opts.addOption("eta0", true, "Initial value of learning rate.");
        opts.addOption("lambda", true, "Regularization parameter.");
        opts.addOption("C", true, "Aggressiveness parameter.");
        opts.addOption("epsilon", true, "Regression tolerance.");
        opts.addOption("r", true, "Aggressiveness parameter.");
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
        int nIter = Integer.parseInt(cliParser.getOptionValue("nIter", "1"));

        if (args.length < 2) {
            System.err.println("Usage: train_file output_base");
            return;
        }
        String trainFile  = args[0];
        String outputBase = args[1];

        boolean isBiasUsed = true;
        FeatureVector datum = new FeatureVector();
        WeightVector weight = new WeightVector(dim);

        LinearLearner learner;
        if (modelType == 0) {
            System.err.println(String.format("Logistic Regressin SGD, dim=%d, lambda=%f, eta0=%f", dim, lambda, eta0));
            learner = new LogRegSGDLearner(0, lambda, new EtaCalculator(eta0, lambda), weight);

        } else if (modelType == 1) {
            System.err.println(String.format("PA Regressin, dim=%d, C=%f, epsilon=%f", dim, C, epsilon));
            learner = new PARegLearner(PARegLearner.PAType.PA1, C, epsilon, weight);

        } else if (modelType == 2) {
            WeightVector sigma = new WeightVector(weight.getDimensions());
            for (int i = 0; i < sigma.getDimensions(); i++) {
                sigma.setValue(i, 1.0f);
            }

            System.err.println(String.format("AROW Regressin, dim=%d, r=%f", dim, r));
            learner = new AROWRegLearner(r, weight, sigma);

        } else {
            System.err.println(String.format("Unknow modelType: %d", modelType));
            return;
        }


        for (int iter = 0; iter < nIter; iter++) {
            BufferedReader trainReader;
            if (trainFile.endsWith(".gz")) {
                trainReader = new BufferedReader(new InputStreamReader(
                                                 new GZIPInputStream(
                                                 new FileInputStream(trainFile))));
            } else {
                trainReader = new BufferedReader(new InputStreamReader(
                                                 new FileInputStream(trainFile)));
            }

            String line;
            while ((line = trainReader.readLine()) != null) {
                SVMLightFormatParser.parse(line, datum, isBiasUsed);
                learner.learn(datum);
            }
            trainReader.close();

            File outputDir = new File(String.format("%s/iter-%05d", outputBase, iter));
            if (! outputDir.exists()) {
                outputDir.mkdirs();
            }

            Configuration conf = new Configuration();
            Path weightPath = new Path(String.format("%s/part-r-00000", outputDir.toString()));
            FileSystem fs = FileSystem.getLocal(conf);

            SequenceFile.Writer weightWriter = new SequenceFile.Writer(fs, conf, weightPath, Text.class, WeightVector.class);
            weightWriter.append(new Text("weight"), learner.getWeight());
            if (modelType == 2) {
                weightWriter.append(new Text("sigma"), ((AROWRegLearner)learner).getSigma());
            }
            weightWriter.close();

        }

    }

}
