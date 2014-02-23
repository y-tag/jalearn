package myorg.examples.util;

import java.io.File;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.Random;

public class RandomSplit {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: out_dir split_num");
            return;
        }
        String outDir = args[0];
        int splitNum  = Integer.parseInt(args[1]);

        if (splitNum <= 0) {
            System.err.println("split_num is le 0: " + splitNum);
            return;
        }

        BufferedWriter writer = null;
        ArrayList<BufferedWriter> bwArray = new ArrayList<BufferedWriter>();
        for (int i = 0; i < splitNum; i++) {
            String outFile = String.format("%s/part-%05d", outDir, i);
            writer = new BufferedWriter(new OutputStreamWriter(
                                        new FileOutputStream(outFile)));
            bwArray.add(writer);
        }

        String line;
        BufferedReader inReader = new BufferedReader(new InputStreamReader(System.in));
        Random rnd = new Random();

        while ((line = inReader.readLine()) != null) {
            int r = rnd.nextInt(bwArray.size());
            bwArray.get(r).write(line);
            bwArray.get(r).write("\n");
        }

        for (int i = 0; i < splitNum; i++) {
            bwArray.get(i).close();
        }
        bwArray.clear();

    }

}
