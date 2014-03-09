package myorg.io;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.Map;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.hadoop.io.Writable;

public class PartedWeightVector implements Writable {
    private int dim = 0;
    private int offset = 0;
    private int size = 0;
    private float[] weightArray = null;

    public PartedWeightVector() {
        this(0, 0, 0);
    }

    public PartedWeightVector(int dim, int offset, int size) {
        this.dim = dim;
        this.offset = offset;
        this.size = size;
        this.weightArray = new float[size];
    }

    public int getDimensions() {
        return dim;
    }

    public int getOffset() {
        return offset;
    }

    public int getSize() {
        return size;
    }

    public float getValue(int index) {
        if (index < offset || offset + size <= index) {
            return 0.0f;
        }
        return weightArray[index - offset];
    }

    public void setValue(int index, float value) {
        if (offset <= index && index < offset + size) {
            weightArray[index - offset] = value;
        }
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(dim);
        out.writeInt(offset);
        out.writeInt(size);
        for (int i = 0; i < size; i++) {
            out.writeFloat(weightArray[i]);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        dim = in.readInt();
        offset = in.readInt();
        size = in.readInt();
        if (size > weightArray.length) {
            weightArray = new float[size];
        }
        for (int i = 0; i < size; i++) {
            weightArray[i] = in.readFloat();
        }
    }

    @Override
    public String toString() {

        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < size; i++) {
            if (weightArray[i] == 0.0f) {
                continue;
            }

            if (sb.length() > 0) {
                sb.append(' ');
            }
            sb.append(i + offset);
            sb.append(':');
            sb.append(weightArray[i]);
        }

        sb.append(" # dim:" + Integer.toString(dim));

        return sb.toString();
    }

}


