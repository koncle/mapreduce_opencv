package com.koncle.mapred;

import com.koncle.blocks.FaceSVM;
import com.koncle.blocks.Utils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Predict {
    public static class PredictMapper
            extends Mapper<Object, Text, IntWritable, IntWritable> {
        private FaceSVM[] svms;
        private int classes = 3;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // 从配置中读取模型保存地址
            Configuration s = context.getConfiguration();
            String[] modelPaths = s.get("models").split("\t");
            svms = new FaceSVM[modelPaths.length];
            for (int i = 0; i < modelPaths.length; ++i) {
                svms[i] = new FaceSVM();
                svms[i].load(modelPaths[i]);
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] strs = value.toString().split("\t");
            int id = Integer.parseInt(strs[0]);
            String claz = strs[1], matStr = strs[2];
            Mat mat = Utils.getMatFromString(matStr);

            // 对测试数据进行预测，
            int[] votes = new int[classes];
            for (int i = 0; i < svms.length; ++i) {
                int pred = (int) svms[i].predict(mat).get(0, 0)[0];
                if (pred == 1) {
                    // 预测类别为svm的类别
                    votes[i] += 1;
                }
            }
            // 获取大多数的结果
            int predLabel = Utils.getMaxIdx(votes);
            // 预测结果以  [class, id] 形式输出
            context.write(new IntWritable(predLabel), new IntWritable(id));
        }
    }

    public static class PredictReducer
            extends Reducer<IntWritable, IntWritable, Text, Text> {
        private Map<Integer, String> classMap;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // 获取类别索引到人名转换的 Map
            String mappingPath = context.getConfiguration().get("classMapping");
            String[] lines = Utils.readHdfs(context.getConfiguration(), mappingPath).split("\n");
            classMap = new HashMap<>();
            for (int i = 0; i < lines.length; ++i) {
                String[] tmp = lines[i].split("\t");
                classMap.put(Integer.parseInt(tmp[1]), tmp[0]);
            }
        }

        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            // 获取该类别对应的人名
            Integer claz = new Integer(key.get());
            String name = classMap.get(claz);
            // 获取该类别的样本 id
            StringBuilder sb = new StringBuilder();
            for (IntWritable id : values) {
                sb.append(id.get());
                sb.append(" ");
            }
            // 删除最后一个 \t
            sb.deleteCharAt(sb.length() - 1);
            // 输出 [人名， 样本id 列表]
            context.write(new Text(name), new Text(sb.toString()));
        }
    }
}
