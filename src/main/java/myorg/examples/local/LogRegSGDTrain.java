package myorg.examples.local;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import myorg.io.FeatureVector;
import myorg.io.WeightVector;
import myorg.util.SVMLightFormatParser;
import myorg.classifier.LogRegSGDLearner;

public class LogRegSGDTrain {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: train_file weight_file");
            return;
        }
        String trainFile  = args[0];
        String weightFile = args[1];

        int dim = 1 << 24;
        float eta0 = 1e-1f;
        float lambda = 1e-7f;
        int numEpochs = 2;
        boolean isBiasUsed = true;
        FeatureVector datum = new FeatureVector();
        WeightVector weight = new WeightVector(dim);


        String line;

        long i = 1;
        for (int n = 0; n < numEpochs; n++) {
            BufferedReader trainReader;
            if (trainFile.endsWith(".gz")) {
                trainReader = new BufferedReader(new InputStreamReader(
                                                 new GZIPInputStream(
                                                 new FileInputStream(trainFile))));
            } else {
                trainReader = new BufferedReader(new InputStreamReader(
                                                 new FileInputStream(trainFile)));
            }

            while ((line = trainReader.readLine()) != null) {
                float eta = eta0 / (1.0f + eta0 * lambda * i);
                SVMLightFormatParser.parse(line, datum, isBiasUsed);
                LogRegSGDLearner.learnWithStochasticOneStep(datum, eta, lambda, weight);
                i++;
            }
            trainReader.close();
        }

        Configuration conf = new Configuration();
        Path weightPath = new Path(weightFile);
        FileSystem fs = FileSystem.getLocal(conf);

        SequenceFile.Writer weightWriter = new SequenceFile.Writer(fs, conf, weightPath, Text.class, WeightVector.class);
        weightWriter.append(new Text(), weight);
        weightWriter.close();
    }

}
