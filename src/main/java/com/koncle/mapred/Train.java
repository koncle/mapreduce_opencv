package com.koncle.mapred;

import com.koncle.blocks.FaceSVM;
import com.koncle.blocks.Utils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.log4j.Logger;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.io.IOException;


public class Train {
    public static class TrainMapper
            extends Mapper<Object, Text, Text, MatWritable> {

        private int classes = 3;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // TODO: 在这里读取从 classes.txt 读取文件
            classes = 3;
        }

        @Override
        protected void map(Object key, Text text, Context context) throws IOException, InterruptedException {
            String[] strings = text.toString().split("\t");
            // 忽略 id
            int sampleClaz = Integer.parseInt(strings[1]);
            Mat mat = Utils.getMatFromString(strings[2].replace("c", "\n"));
            // 将当前前样本标记不同的类别，发送给不同的SVM
            // 如 类别为 2， 则 svm0 为 0， svm1 为 10， svm2 为 1
            for (int i = 0; i < classes; ++i) {
                if (sampleClaz != i) {
                    context.write(new Text(i + "\t" + 0), new MatWritable(mat));
                } else {
                    context.write(new Text(i + "\t" + 1), new MatWritable(mat));
                }
            }
        }
    }

    public static class TrainPartitioner
            extends HashPartitioner<Text, MatWritable> {
        @Override
        public int getPartition(Text text, MatWritable matWritable, int i) {
            // 按训练数据类别来排序
            String s = text.toString().split("\t")[0];
            return super.getPartition(new Text(s), matWritable, i);
        }
    }

    public static class TrainReducer
            extends Reducer<Text, MatWritable, IntWritable, Text> {
        FaceSVM[] svms = null;
        int classes = 3;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            svms = new FaceSVM[classes];
            for (int i = 0; i < classes; ++i) {
                svms[i] = new FaceSVM();
            }

            // 读取模型保存地址
            Configuration s = context.getConfiguration();
            modelPaths = s.get("models").split("\t");
        }

        Mat labels = null;
        Mat trainData = null;
        int lastSvmIndex = -1;
        FaceSVM currentSVM = null;
        String[] modelPaths = null;

        @Override
        protected void reduce(Text key, Iterable<MatWritable> values, Context context) throws IOException, InterruptedException {
            String[] strs = key.toString().split("\t");
            int svmIndex = Integer.parseInt(strs[0]);
            int label = Integer.parseInt(strs[1]);

            if (lastSvmIndex != svmIndex) {
                // train svm
                if (this.currentSVM != null) {
                    // TrainData trainData = TrainData.create(this.trainData, Ml.ROW_SAMPLE, this.labels);
                    // 得到所有训练数据，即可训练模型，无需增量式更新
                    this.currentSVM.train(trainData, labels);
                }
                // refresh data
                this.currentSVM = svms[svmIndex];
                labels = new Mat();
                trainData = new Mat();
                lastSvmIndex = svmIndex;
            }
            // save data to train
            for (MatWritable value : values) {
                this.labels.push_back(new Mat(1, 1, CvType.CV_32S, new Scalar(label)));
                this.trainData.push_back(value.getMat());
            }
        }

        Logger logger = Logger.getLogger(this.getClass());

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // 训练剩下的数据
            if (this.currentSVM != null) {
                this.currentSVM.train(this.trainData, this.labels);
            }
            // 保存svm
            // 由于 opencv的 SVM 没有提供输出模型字符串之类的接口
            // 只能先写入本地文件，再进行读取，保存到 hdfs 中
            // TODO: 需要优化，保存文件太耗时
            for (int i = 0; i < classes; ++i) {
                String path = modelPaths[i];
                logger.info("Saving model " + path + ".............");
                svms[i].save(path);
                String json = Utils.readFile(path);
                Utils.deleteFile(path);
                Utils.writeHdfs(context.getConfiguration(), json, path);
                // context.write(new IntWritable(i), new Text(json));
            }
        }
    }
}
