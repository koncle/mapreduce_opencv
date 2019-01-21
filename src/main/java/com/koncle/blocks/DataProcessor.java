package com.koncle.blocks;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.TrainData;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static com.koncle.blocks.Utils.println;

public class DataProcessor {

    /*
     * 读入图片，进行旋转和缩放，PCA进行投影，reshape成(1*xxx)
     * */
    public static Mat preprocess(String oneImagePath, int pca_dim) {
        Mat img = readImage(oneImagePath);
        Mat processed_img = pca(img, pca_dim);
        return processed_img.reshape(1, 1);
        // Utils.println(processed_img.type() + " " + res.type() + " " + img.type());
        // res = new Mat();
        // processed_img.convertTo(res, 0);
    }

    public static Mat readImage(String path) {
        try {
            // TODO: 用 hadoop jar 运行，这里会报 task failed 不知道为什么
            return Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
        } catch (Exception e) {
            println("Failed???");
            println(e.toString());
            e.printStackTrace();
        }
        return null;
    }

    public static Mat imgAugmentation(Mat img) {
        img = rotateAndScale(img, 30, 1.);
        img = flip(img, 0);
        return img;
    }

    /**
     * @param img
     * @param angle 旋转角度，如 30.
     * @param scale 缩放大小，如 1.5
     * @return
     */
    public static Mat rotateAndScale(Mat img, double angle, double scale) {
        Point center = new Point(img.width() / 2., img.height() / 2.);
        Mat affineTrans = Imgproc.getRotationMatrix2D(center, angle, scale);
        Mat dst = new Mat();
        Imgproc.warpAffine(img, dst, affineTrans, dst.size(), Imgproc.INTER_NEAREST);
        return dst;
    }

    /*
     * flipCode
     * = 0 向下翻转
     * > 0 向右翻转
     * < 0 向右向下翻转
     * */
    public static Mat flip(Mat img, int flipCode) {
        Mat dst = new Mat();
        Core.flip(img, dst, flipCode);
        return dst;
    }

    public static Mat pca(Mat img, int toDim) {
        // 去中心化
        Mat mean_img = new Mat();
        Core.subtract(img, new Mat(img.size(), img.type(), Core.mean(img)), mean_img);
        // 降维
        Mat mean = new Mat();
        Mat vectors = new Mat();
        Core.PCACompute(mean_img, mean, vectors, toDim);
        // 返回前n个向量为降维后
        // Utils.println(vectors.size());
        return vectors;
    }

    public static void main(String[] args) {
    }
}
