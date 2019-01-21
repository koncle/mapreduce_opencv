package com.koncle.mapred;

import com.koncle.blocks.Utils;
import org.apache.hadoop.io.Writable;
import org.opencv.core.Mat;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class MatWritable implements Writable {
    private Mat mat;
    private int t;

    public MatWritable() {
    }

    public MatWritable(Mat mat) {
        this.mat = mat;
    }

    int maxlength = 20000;

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        String s = this.mat.dump();
        byte[] bytes = s.getBytes();
        dataOutput.writeInt(bytes.length);
        dataOutput.write(bytes);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        int length = dataInput.readInt();
        byte[] bytes = new byte[length];
        dataInput.readFully(bytes);
        String matString = new String(bytes);
        this.mat = Utils.getMatFromString(matString);
    }

    public Mat getMat() {
        return mat;
    }

    public void setMat(Mat mat) {
        this.mat = mat;
    }
}
