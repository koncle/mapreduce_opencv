package com.koncle.mapred;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;
import org.opencv.core.Core;

import java.io.IOException;

public class Main {
    static Logger logger = Logger.getLogger(Main.class);

    public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        int stage = 1;

        Configuration conf;
        Job job;
        if (stage <= 1) {
            // configuration file
            conf = new Configuration();
            job = Job.getInstance(conf, "PreProcess train data");
            job.setJarByClass(PreProcess.class);
            job.setMapperClass(PreProcess.PreprocessMapper.class);
            job.setReducerClass(PreProcess.PreprocessReducer.class);
            job.setPartitionerClass(PreProcess.PreprocessPartitioner.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(MatWritable.class);
            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[1]));
            job.waitForCompletion(true);
        }

        if (stage <= 2) {
            conf = new Configuration();
            job = Job.getInstance(conf, "Preprocess test data");
            job.setJarByClass(PreProcess.class);
            job.setMapperClass(PreProcess.PreprocessMapper.class);
            job.setReducerClass(PreProcess.PreprocessReducer.class);
            job.setPartitionerClass(PreProcess.PreprocessPartitioner.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(MatWritable.class);
            FileInputFormat.addInputPath(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[2]));
            job.waitForCompletion(true);
        }

        if (stage <= 3) {
            Configuration conf2 = new Configuration();
            conf2.set("models", "path0.json\tpath1.json\tpath2.json");

            job = Job.getInstance(conf2, "Train");
            job.setJarByClass(Train.class);
            job.setMapperClass(Train.TrainMapper.class);
            job.setReducerClass(Train.TrainReducer.class);
            job.setPartitionerClass(Train.TrainPartitioner.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(MatWritable.class);
            FileInputFormat.addInputPath(job, new Path(args[1]));
            FileOutputFormat.setOutputPath(job, new Path(args[3]));
            job.waitForCompletion(true);
        }

        if (stage <= 4) {
            Configuration conf3 = new Configuration();

            // 设置两个变量

            // 让模型保存路径
            conf3.set("models", "path0.json\tpath1.json\tpath2.json");
            // 以让 Predict 知道在哪里找到 类别与人名映射文件
            conf3.set("classMapping", "classMapping.txt");

            job = Job.getInstance(conf3, "Predict");
            job.setJarByClass(Predict.class);
            job.setMapperClass(Predict.PredictMapper.class);
            job.setReducerClass(Predict.PredictReducer.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.addInputPath(job, new Path(args[2]));
            FileOutputFormat.setOutputPath(job, new Path(args[4]));
            System.exit(job.waitForCompletion(true) ? 1 : 0);
        }
    }
}
