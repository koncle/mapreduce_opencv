package com.koncle.blocks;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * 给定图片文件夹，按照文件夹里面的文件夹名字创建对应类别，
 */
public class ImageFolder {
    private String folder;
    // 保存类别名，及其对应的图片路径
    private Map<String, List<String>> classWithImagePathMap = new HashMap<>();
    // 类别名(String类型)对应的标签(int类型)
    private Map<String, Integer> classLabelMap = new HashMap<>();
    private int size = 0;
    private int classes = -1;

    public ImageFolder(String path, int classes) {
        this.folder = path;
        this.classes = classes;
        retrieveImages(10);
    }

    private void retrieveImages(int minNum) {
        File imageFolder = new File(this.folder);
        int classes = 0;
        if (imageFolder.exists()) {
            for (File folder : imageFolder.listFiles()) {
                List<String> filenames = new ArrayList<String>();
                assert folder.isDirectory();
                int num = 0;
                for (File img : folder.listFiles()) {
                    filenames.add(img.getAbsolutePath());
                    this.size += 1;
                    num += 1;
                }
                if (num > minNum) {
                    this.classWithImagePathMap.put(folder.getName(), filenames);
                    this.classLabelMap.put(folder.getName(), classes++);
                    if (this.classes == classes) break;
                }
            }
        } else {
            Utils.println("No Such image folder!");
        }
        this.classes = this.classWithImagePathMap.size();
    }

    public void writeFile(String outputFile) {
        int count = 0;
        File f = new File(outputFile);
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            for (String name : this.classWithImagePathMap.keySet()) {
                List<String> l = this.classWithImagePathMap.get(name);
                // TODO: 可以在这里划分训练集与测试集，分打开两个文件分别进行操作
                for (String path : l) {
                    // TODO: 如果将图片传入HDFS中， path需要更换为 HDFS 的 path
                    bw.write(count + "\t" + this.classLabelMap.get(name) + "\t" + path + "\n");
                    count += 1;
                }
            }
            bw.flush();
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeClassMapping(String classOutput) {
        File f = new File(classOutput);
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(f));
            for (String name : this.classLabelMap.keySet()) {
                int claz = this.classLabelMap.get(name);
                bw.write(name + "\t" + claz + "\n");
            }
            bw.flush();
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeClasses(String output) {
        Utils.writeFile(this.classes + "", output);
    }

    public TrainData generateTrainData(int classes) {
        Set<List<String>> imgPaths = getAllImagePaths();
        Mat data = new Mat();
        Map<String, Integer> imgLabelMap = getImageLabelMap();
        List<Integer> labelList = new ArrayList<>();
        int count = classes;
        for (List<String> imgPath : imgPaths) {
            if (count > -1) {
                count--;
            }
            for (String p : imgPath) {
                data.push_back(DataProcessor.preprocess(p, 10));
                String class_name = Utils.getClassNameFromPath(p);
                int label = imgLabelMap.get(class_name);
                labelList.add(label);
            }
            if (count == 0) break;
        }
        Mat labels = new Mat(labelList.size(), 1, CvType.CV_32S);
        for (int i = 0; i < labelList.size(); ++i) {
            labels.put(i, 0, labelList.get(i));
        }
        return TrainData.create(data, Ml.ROW_SAMPLE, labels);
    }

    public String getAImage() {
        return this.classWithImagePathMap.values().iterator().next().get(0);
    }

    public Set<String> getNames() {
        return this.classWithImagePathMap.keySet();
    }

    public Set<List<String>> getAllImagePaths() {
        Set<List<String>> set = new HashSet<List<String>>();
        for (List l : this.classWithImagePathMap.values()) {
            set.add(l);
        }
        return set;
    }

    public int getTotalImageSize() {
        return this.size;
    }

    public Map<String, List<String>> getImageNameMap() {
        return this.classWithImagePathMap;
    }

    public Map<String, Integer> getImageLabelMap() {
        return this.classLabelMap;
    }

    public int getClases() {
        return this.classes;
    }
}
