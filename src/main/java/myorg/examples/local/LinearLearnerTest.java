package myorg.examples.local;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;
import java.util.Collections;
import java.util.ArrayList;

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
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.util.ScoreStruct;
import myorg.util.AUCCalculator;
import myorg.util.SVMLightFormatParser;
import myorg.classifier.LogRegSGDLearner;

public class LinearLearnerTest {

    public static void main(String[] args) throws Exception {

        Options opts = new Options();
        opts.addOption("logistic", false, "Use logistic function for prediction.");

        CommandLine cliParser = new GnuParser().parse(opts, args);
        args = cliParser.getArgs();

        boolean isLogisticUsed = cliParser.hasOption("logistic");

        if (args.length < 2) {
            System.err.println("Usage: test_file weight_file");
            return;
        }
        String testFile  = args[0];
        String weightFile = args[1];

        boolean isBiasUsed = true;
        FeatureVector datum = new FeatureVector();
        WeightVector weight = new WeightVector();

        Configuration conf = new Configuration();
        Path weightPath = new Path(weightFile);
        FileSystem fs = FileSystem.getLocal(conf);

        SequenceFile.Reader reader = new SequenceFile.Reader(fs, weightPath, conf);
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
        reader.close();


        String line;

        long i = 1;
        BufferedReader testReader;
        if (testFile.endsWith(".gz")) {
            testReader = new BufferedReader(new InputStreamReader(
                                            new GZIPInputStream(
                                            new FileInputStream(testFile))));
        } else {
            testReader = new BufferedReader(new InputStreamReader(
                                            new FileInputStream(testFile)));
        }

        ArrayList<ScoreStruct> ssList = new ArrayList<ScoreStruct>();

        while ((line = testReader.readLine()) != null) {
            SVMLightFormatParser.parse(line, datum, isBiasUsed);

            float prediction = weight.innerProduct(datum);
            if (isLogisticUsed) {
                prediction = 1.0f / (1.0f + (float)Math.exp(-prediction));
            }

            ScoreStruct ss = new ScoreStruct();
            ss.value = prediction;
            if (datum.getLabel() > 0) { ss.positive = 1.0f; }
            else                      { ss.negative = 1.0f; }
            ssList.add(ss);

            i++;
        }
        testReader.close();

        Collections.sort(ssList);

        double curAUROC = 0.0;
        double curAUPRC = 0.0;
        double curRMSE = 0.0;
        double curMAE = 0.0;
        double curNLL = 0.0;

        double oldPosSum = 0.0;
        double oldPrecision = 0.0;
        double posNum = 0.0;
        double posSum = 0.0;
        double negNum = 0.0;
        double negSum = 0.0;

        float lastPredict = Float.MAX_VALUE;

        for (ScoreStruct ss : ssList) {
            if (lastPredict != ss.value && (posSum + negSum) > 0.0) {
                curAUROC += (oldPosSum + posSum) * negNum / 2.0;
                oldPosSum = posSum;
                negNum = 0;

                double precision = posSum / (posSum + negSum);
                curAUPRC += (oldPrecision + precision) * posNum / 2.0;
                oldPrecision = precision;
                posNum = 0.0;
            }

            lastPredict = ss.value;
            posNum += ss.positive;
            posSum += ss.positive;
            negNum += ss.negative;
            negSum += ss.negative;

            double y = (ss.positive > 0.0) ? 1.0 : 0.0;
            curRMSE += Math.pow(y - ss.value, 2.0);
            curMAE += Math.abs(y - ss.value);

            double p = Math.max(Math.min(ss.value, 1.0 - 1.0e-15), 1.0e-15);
            double tmp = (ss.positive > 0) ? p : (1.0 - p);
            curNLL -= Math.log(tmp);
        }

        System.out.println(String.format("%s\t%d", "Number of samples", (long)(posSum + negSum)));
        System.out.println(String.format("%s\t%d", "Number of positive samples", (long)posSum));
        System.out.println(String.format("%s\t%d", "Number of negative samples", (long)negSum));

        curAUROC += (oldPosSum + posSum) * negNum / 2.0;
        curAUROC /= (posSum * negSum);
        System.out.println(String.format("%s\t%.7f", "AUC", curAUROC));

        double precision = posSum / (posSum + negSum);
        curAUPRC += (oldPrecision + precision) * posNum / 2.0;
        curAUPRC /= posSum;
        System.out.println(String.format("%s\t%.7f", "AUPRC", curAUPRC));

        curRMSE = Math.sqrt(curRMSE / (posSum + negSum));
        System.out.println(String.format("%s\t%.7f", "RMSE", curRMSE));
        
        curMAE = curMAE / (posSum + negSum);
        System.out.println(String.format("%s\t%.7f", "MAE", curMAE));

        System.out.println(String.format("%s\t%.7f", "NLL", curNLL));
    }

}
