package com.koncle.blocks;

import org.opencv.core.*;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.SVMSGD;
import org.opencv.ml.TrainData;

import static org.opencv.core.CvType.*;

public class SVMTest {

    // SGD支持向量机
    public static Mat MySvmsgd(Mat trainingData, Mat labels, Mat testData) {
        SVMSGD Svmsgd = SVMSGD.create();
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        Svmsgd.setTermCriteria(criteria);

        Svmsgd.setInitialStepSize(2);
        Svmsgd.setSvmsgdType(SVMSGD.SGD);
        Svmsgd.setMarginRegularization(0.5f);

        // TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        // System.out.println("td responses: " + td.getResponses().dump());
        // boolean success = Svmsgd.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        boolean success = Svmsgd.train(trainingData, Ml.ROW_SAMPLE, labels);
        System.out.println("SVMSGD training result: " + success);
        // svm.save("D:/bp.xml");//存储模型
        // svm.load("D:/bp.xml");//读取模型

        Mat responseMat = new Mat();
        Svmsgd.predict(testData, responseMat, 0);
        System.out.println("SVM_SGD responseMat:\n" + responseMat.dump());
        for (int i = 0; i < responseMat.height(); i++) {
            if (responseMat.get(i, 0)[0] == 0)
                System.out.println("Boy\n");
            if (responseMat.get(i, 0)[0] == 1)
                System.out.println("Girl\n");
        }
        return responseMat;
    }

    // 支持向量机
    public static Mat MySvm(Mat trainingData, Mat labels, Mat testData) {
        SVM svm = SVM.create();
        svm.setKernel(SVM.RBF);
        svm.setType(SVM.C_SVC);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        svm.setTermCriteria(criteria);

        svm.setGamma(0.5);
        svm.setNu(0.5);
        svm.setC(1);

        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);

        boolean success = svm.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("Svm training reaaasult: " + success);

        Mat responseMat = new Mat();
        svm.predict(testData, responseMat, 0);
        System.out.println("SVM responseMat:\n" + responseMat.dump());
        for (int i = 0; i < responseMat.height(); i++) {
            if (responseMat.get(i, 0)[0] == 0)
                System.out.println("Boy\n");
            if (responseMat.get(i, 0)[0] == 1)
                System.out.println("Girl\n");
        }
        return responseMat;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        int[][] data = {
                {2, 3},
                {3, 4},
                {4, 9},
                {4, 5},
                {4, 5},
                {4, 5}
        };

        Mat dataMat = new Mat(data.length, 2, CvType.CV_32FC1);
        for (int i = 0; i < data.length; ++i) {
            for (int j = 0; j < 2; ++j) {
                dataMat.put(i, j, data[i][j]);
            }
        }

        int[] classes = {1, 0, 1, 0, 0, 0};
        Mat labels = new Mat(classes.length, 1, CV_32S);
        for (int i = 0; i < classes.length; ++i) {
            labels.put(i, 0, classes[i]);
        }
        Utils.println(dataMat.dump());
        Utils.println(labels.dump());

        // 输入 CV_FC1 + CV_FC1, same type
        MySvmsgd(dataMat, labels, dataMat);

        // 输入 CV_FC1 + CV_32S
        // MySvm(dataMat, labels, dataMat);
    }
}
