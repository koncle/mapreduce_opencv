package com.koncle.blocks;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.ml.TrainData;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    static String imagePath = "/home/koncle/lfw_data/mini_lfw";

    public static void testSVM() {
        ImageFolder imageFolder = new ImageFolder(imagePath, 3);
        TrainData trainData = imageFolder.generateTrainData(3);
        Mat data = trainData.getSamples();
        Mat labels = trainData.getResponses();

        Utils.println(data.size());
        Utils.println(labels.size());
        Utils.println(labels.dump());

        FaceSVM svm = new FaceSVM();
        svm.train(data, labels);
//        Utils.println(svm.predict(Mat.ones(new Size(2500, 1), CvType.CV_32FC1)).dump());
//        Utils.println("calculate F1 :");
//        Utils.println(Utils.F1(svm.predict(data), labels, imageFolder.getClases()));

        svm.save("what.json");
        // 测试读写
        svm.load("what2.json");
        Utils.println(svm.predict(data).dump());
        Utils.println(Utils.F1(svm.predict(data), labels, imageFolder.getClases()));
    }

    // 为local hadoop 创建图片索引文件
    public static void createImageIndex() {
        ImageFolder imageFolder = new ImageFolder(imagePath, 3);
        // 索引文件内的图片地址为本地图片的地址，不是 HDFS 的地址，可能需要改成 HDFS 的地址
        imageFolder.writeFile("imagePath.txt");
        imageFolder.writeClassMapping("classMapping.txt");
        imageFolder.writeFile("classes.txt");
    }

    public static void main(String[] args) {
        createImageIndex();
        // testSVM();
    }
}
