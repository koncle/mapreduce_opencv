package com.koncle.mapred;

import com.koncle.blocks.DataProcessor;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.opencv.core.Mat;

import java.io.IOException;

public class PreProcess {
    public static class PreprocessMapper
            extends Mapper<Object, Text, Text, MatWritable> {

        @Override
        public void map(Object o, Text text, Context context) throws IOException, InterruptedException {
            // 测试是否能够在hadoop中保存文件 => 可以的
            // Utils.writeFile("test", "/home/koncle/asdfasdfasdfasdf/src/main/java/com/koncle/mapred/hhhh.txt");

            // 读取 imagePath.txt 文件的一行
            String[] s = text.toString().split("\t");
            String id = s[0];
            String path = s[2];
            String claz = s[1];

            // 读取图片进行处理
            Mat img = DataProcessor.readImage(path);
            Mat augImag = DataProcessor.flip(img, -1);
            augImag = DataProcessor.rotateAndScale(augImag, 30, 1.);
            augImag = DataProcessor.pca(augImag, 100).reshape(1, 1);

            // 输出一个随机的字符来打乱图片
            context.write(new Text((int) (Math.random() * 1000) + "\t" + claz + "\t" + id), new MatWritable(augImag));
        }
    }

    public static class PreprocessPartitioner
            extends HashPartitioner<Text, MatWritable> {
        @Override
        public int getPartition(Text text, MatWritable matWritable, int i) {
            // 获取随机字符串作为分割依据
            String s = text.toString().split("\t")[0];
            return super.getPartition(new Text(s), matWritable, i);
        }
    }

    public static class PreprocessReducer extends Reducer<Text, MatWritable, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<MatWritable> values, Context context) throws IOException, InterruptedException {
            // 获取真实 类别和id
            String[] tmp = key.toString().split("\t");
            String claz = tmp[1];
            String id = tmp[2];

            // 将特征输出
            for (MatWritable mat : values) {
                // 将 mat输出的字符串中的 \n 换成 c，以便于存储在一行中
                context.write(new Text(id + "\t" + claz), new Text(mat.getMat().dump().replaceAll("\n", "c")));
            }
        }
    }
}
