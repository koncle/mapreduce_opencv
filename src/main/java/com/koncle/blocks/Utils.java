package com.koncle.blocks;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Utils {
    public static <T> void println(T t) {
        System.out.println(t);
    }

    public static <T> void print(T t) {
        System.out.print(t);
    }

    public static List<String> getFiles(String path) {
        List filenames = new ArrayList<String>();
        File file = new File(path);
        if (file.exists()) {
            // 目录
            if (file.isDirectory()) {
                File[] files = file.listFiles();
                for (File f : files) {
                    // 递归遍历文件
                    if (f.isDirectory()) {
                        List next_filenames = getFiles(f.getAbsolutePath());
                        filenames.addAll(next_filenames);
                    } else {
                        filenames.add(f.getAbsolutePath());
                    }
                }
            } else {
                // 文件
                filenames.add(file.getAbsolutePath());
            }
        }
        return filenames;
    }

    public static String readFile(String path) {
        File f = new File(path);
        try {
            InputStreamReader is = new InputStreamReader(new FileInputStream(f));
            BufferedReader br = new BufferedReader(is);
            String line = br.readLine();
            String s = "";
            while (line != null) {
                s += line + "\n";
                line = br.readLine();
            }
            br.close();
            is.close();
            return s;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static boolean writeFile(String s, String path) {
        File f = new File(path);
        boolean sucess = true;
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            bw.write(s);
            bw.flush();
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
            sucess = false;
        }
        return sucess;
    }

    public static boolean deleteFile(String path) {
        File f = new File(path);
        if (f.exists()) {
            return f.delete();
        } else {
            return true;
        }
    }

    public static boolean writeHdfs(Configuration conf, String json, String path) {
        boolean sucess = true;
        Path hdfsPath = new Path(path);
        try {
            FSDataOutputStream outputStream = FileSystem.get(conf).create(hdfsPath);
            OutputStreamWriter osw = new OutputStreamWriter(outputStream);
            BufferedWriter bw = new BufferedWriter(osw);
            bw.write(json);
            bw.flush();
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
            sucess = false;
        }
        return sucess;
    }

    public static String readHdfs(Configuration conf, String path) {
        Path hdfsPath = new Path(path);
        String res = "";

        try {
            FSDataInputStream inputStream = FileSystem.get(conf).open(hdfsPath);
            InputStreamReader is = new InputStreamReader(inputStream);
            BufferedReader br = new BufferedReader(is);
            String line = null;
            line = br.readLine();
            while (line != null) {
                res += line + "\n";
                line = br.readLine();
            }
            br.close();
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return res;
    }

    public static boolean saveMat(Mat mat, String path) {
        String s = mat.dump();
        return writeFile(s, path);
    }

    public static Mat loadMat(String path) {
        String s = readFile(path);
        return getMatFromString(s);
    }

    public static Mat getMatFromString(String s) {
        if (s == null) return null;

        List<double[]> l = new ArrayList<>();
        String[] strings = s.split("\n");
        for (String line : strings) {
            // println(line);
            line = line.substring(1, line.length() - 1);
            double[] array = getArrayFromString(line.trim().split(","));
            l.add(array);
        }

        if (l.size() > 1) {
            assert l.get(0).length == l.get(1).length;
        }

        int row = l.size();
        int col = l.get(0).length;
        Mat mat = new Mat(row, col, CvType.CV_32FC1);
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                mat.put(i, j, l.get(i)[j]);
            }
        }
        return mat;
    }

    private static double[] getArrayFromString(String[] strings) {
        double[] array = new double[strings.length];
        for (int i = 0; i < strings.length; ++i) {
            array[i] = Double.parseDouble(strings[i].trim());
        }
        return array;
    }

    public static int getMaxIdx(int[] array) {
        int maxIdx = -1, maxValue = -1;
        for (int i = 0; i < array.length; ++i) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public static void imShow(Mat img) {
        HighGui.imshow("show", img);
        HighGui.waitKey(0);
    }

    public static String getClassNameFromPath(String path) {
        String[] s = path.split("\\.")[0].split("/");
        return s[s.length - 2];
    }

    public static double F1(Mat predicts, Mat groundtruth, int classes) {
        double[][] confusionMatrix = getConfusionMatrix(predicts, groundtruth, classes);

        double[] TP = new double[classes];
        for (int i = 0; i < classes; ++i) {
            TP[i] = confusionMatrix[i][i];
        }
        double[] FP = new double[classes];
        for (int i = 0; i < classes; ++i) {
            for (int j = 0; j < classes; ++j) {
                if (i == j) continue;
                FP[i] += confusionMatrix[i][j];
            }
        }
        double[] FN = new double[classes];
        for (int i = 0; i < classes; ++i) {
            for (int j = 0; j < classes; ++j) {
                if (i == j) continue;
                FN[i] += confusionMatrix[j][i];
            }
        }
        double micro_f1 = getMicroF1(TP, FP, FN);
        double macro_f1 = getMacroF1(TP, FP, FN);
        return micro_f1;
    }

    public static double[][] getConfusionMatrix(Mat predicts, Mat groundtruth, int classes) {
        double[][] confusionMatrix = new double[classes][classes];
        for (int i = 0; i < groundtruth.rows(); ++i) {
            int gt = (int) groundtruth.get(i, 0)[0];
            int pred = (int) predicts.get(i, 0)[0];
            confusionMatrix[pred][gt] += 1;
        }
        return confusionMatrix;
    }


    public static double getMacroF1(double[] TP, double[] FP, double[] FN) {
        double precision = 0.;
        double recall = 0.;
        for (int i = 0; i < TP.length; ++i) {
            if (!(TP[i] == 0 && FP[i] == 0)) {
                precision += TP[i] / (TP[i] + FP[i]);
            } else {
                precision += 1;
            }
            if (!(TP[i] == 0 && FN[i] == 0)) {
                recall += TP[i] / (TP[i] + FN[i]);
            } else {
                recall += 1;
            }
        }
        precision /= TP.length;
        recall /= TP.length;
        return 2 * (recall * precision) / (recall + precision);
    }

    public static double getMicroF1(double[] TP, double[] FP, double[] FN) {
        double precision_numerator = 0.;
        double precision_denominator = 0.;
        double recall_denominator = 0.;

        for (int i = 0; i < TP.length; ++i) {
            precision_numerator += TP[i];
            precision_denominator += TP[i] + FP[i];
            recall_denominator += TP[i] + FN[i];
        }
        double precision = precision_numerator / precision_denominator;
        double recall = precision_numerator / recall_denominator;
        return 2 * (recall * precision) / (recall + precision);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // Test F1 score
        Mat pred = new Mat(5, 1, CvType.CV_32FC1);
        Mat gt = new Mat(5, 1, CvType.CV_32FC1);
        int[] a = {0, 1, 2, 2, 0};
        int[] b = {0, 0, 1, 2, 0};
        for (int i = 0; i < a.length; ++i) {
            pred.put(i, 0, a[i]);
            gt.put(i, 0, b[i]);
        }
        println(F1(pred, gt, 3));

        // Test save and load mat
        Mat data = Mat.diag(pred);
        String savePath = "mat.txt";

        println(data.dump());
        Utils.saveMat(data, savePath);

        Mat mat = Utils.loadMat(savePath);
        println(mat.dump());
    }
}
