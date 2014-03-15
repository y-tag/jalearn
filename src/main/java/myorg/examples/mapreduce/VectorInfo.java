package myorg.examples.mapreduce;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

public class VectorInfo implements WritableComparable<VectorInfo> {

    private int taskId;
    private String vectorType;

    public VectorInfo() {
        this(0, "");
    }

    public VectorInfo(int taskId, String vectorType) {
        this.taskId = taskId;
        this.vectorType = vectorType;
    }

    public int getTaskId() {
        return taskId;
    }

    public VectorInfo setTaskId(int taskId) {
        this.taskId = taskId;
        return this;
    }

    public String getVectorType() {
        return vectorType;
    }

    public VectorInfo setVectorType(String vectorType) {
        this.vectorType = vectorType;
        return this;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(taskId);
        out.writeUTF(vectorType);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        taskId = in.readInt();
        vectorType = in.readUTF();
    }

    @Override
    public int compareTo(VectorInfo vi) {
        int thisTaskId = this.taskId;
        int thatTaskId = vi.getTaskId();

        if (thisTaskId == thatTaskId) {
            String thisVectorType = this.vectorType;
            String thatVectorType = vi.getVectorType();

            if (thisVectorType.equals(thatVectorType)) {
                return 0;
            } else {
                return thisVectorType.compareTo(thatVectorType);
            }

        } else {
            return (thisTaskId < thatTaskId) ? -1 : +1;
        }
    }

}
