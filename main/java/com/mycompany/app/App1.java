package com.mycompany.app;

import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class App1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		 String path = "/opt/spark/spark-2.2.1-bin-hadoop2.7/data/mllib/sample_multiclass_classification_data.txt";    	// on charge spark COntext
    	SparkConf sparkConf = new SparkConf();
		sparkConf.setMaster("local[*]");
		sparkConf.setAppName("Nasdaq prediction	");
    	
		SparkSession spark = SparkSession
				.builder()
				.appName("JavaMultilayerPerceptronClassifierExample").config(sparkConf)
				.getOrCreate();
			    // $example on$
			    // Load training data
			    //String path = "data/mllib/sample_multiclass_classification_data.txt";
			    Dataset<Row> dataFrame = spark.read().format("libsvm").load(path);
			    dataFrame.show();
			    // Split the data into train and test
			    Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
			    Dataset<Row> train = splits[0];
			    Dataset<Row> test = splits[1];

			    // specify layers for the neural network:
			    // input layer of size 4 (features), two intermediate of size 5 and 4
			    // and output of size 3 (classes)
			    int[] layers = new int[] {4,64, 32, 16, 3};

			    // create the trainer and set its parameters
			    MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
			      .setLayers(layers)
			      .setBlockSize(128)
			      .setSeed(1234L)
			      .setMaxIter(100);

			    // train the model
			    MultilayerPerceptronClassificationModel model = trainer.fit(train);
			    
			    // compute accuracy on the test set
			    Dataset<Row> result = model.transform(test);
			    Dataset<Row> predictionAndLabels = result.select("prediction", "label");
			    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
			      .setMetricName("accuracy");

			    System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
			    dataFrame.show(false);
			    // $example off$

			    spark.stop();
	}

}
