package myorg.examples.local;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;
import java.util.ArrayList;

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
            prediction = 1.0f / (1.0f + (float)Math.exp(-prediction));

            ScoreStruct ss = new ScoreStruct();
            ss.value = prediction;
            if (datum.getLabel() > 0) { ss.positive = 1.0f; }
            else                      { ss.negative = 1.0f; }
            ssList.add(ss);

            i++;
        }
        testReader.close();

        double auc = AUCCalculator.calcAUC(ssList);

        System.out.println(auc);

    }

}
