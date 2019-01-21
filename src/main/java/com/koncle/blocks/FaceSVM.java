package com.koncle.blocks;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;

import static org.opencv.core.CvType.CV_32S;

public class FaceSVM {
    private SVM svm;

    public FaceSVM() {
        // TODO: 使用 hadoop jar 命令进行执行时候，无法创建新的 SVM
        svm = SVM.create();
        svm.setKernel(SVM.RBF);
        svm.setType(SVM.C_SVC);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        svm.setTermCriteria(criteria);
        svm.setGamma(0.5);
        svm.setNu(0.5);
        svm.setC(1);
    }

    public void save(String path) {
        svm.save(path);
    }

    public void load(String path) {
        this.svm = SVM.load(path);
    }

    public boolean train(Mat data, Mat label) {
        TrainData td = TrainData.create(data, Ml.ROW_SAMPLE, label);
        boolean success = svm.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        return success;
        // Utils.println("Svm training result: " + success);
    }

    public Mat predict(Mat data) {
        Mat responseMat = new Mat();
        svm.predict(data, responseMat, 0);
        // Utils.println("SVM responseMat:\n" + responseMat.dump());
        return responseMat;
    }

    // VM Options :   -Djava.library.path=/home/koncle/opencv/build/lib
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        int[][] data = {
                {2, 3},
                {3, 4},
                {4, 9},
                {4, 5}
        };

        Mat dataMat = new Mat(4, 2, CvType.CV_32FC1);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                dataMat.put(i, j, data[i][j]);
            }
        }

        int[] classes = {1, 0, 1, 0};
        Mat labels = new Mat(4, 1, CV_32S);
        for (int i = 0; i < 4; ++i) {
            labels.put(i, 0, classes[i]);
        }

        FaceSVM svm = new FaceSVM();
        svm.train(dataMat, labels);
        Utils.println(svm.predict(dataMat).dump());
    }
}
