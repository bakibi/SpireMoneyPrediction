package com.mycompany.app;



import java.util.ArrayList;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vectors;
import static org.apache.spark.ml.linalg.SQLDataTypes.*;
import static org.apache.spark.sql.types.DataTypes.*;
public class App 
{
    public static void main( String[] args )
    {
    	 //String path = "/opt/spark/spark-2.2.1-bin-hadoop2.7/data/mllib/sample_multiclass_classification_data.txt";    	// on charge spark COntext
    	SparkConf sparkConf = new SparkConf();
		sparkConf.setMaster("local[*]");
		sparkConf.setAppName("Nasdaq prediction	");
    	
		SparkSession spark = SparkSession
				.builder()
				.appName("JavaMultilayerPerceptronClassifierExample").config(sparkConf)
				.getOrCreate();
		
		// on charge les donnees 
    	//String path = "/opt/spark/spark-2.2.1-bin-hadoop2.7/data/mllib/sample_multiclass_classification_data.txt";
		String path = "stocks.csv";
    	Dataset<Row> dataFrame = spark.read().format("com.databricks.spark.csv").option("header", true).load(path);
    	
    	
    	StructType schema = createStructType(new StructField[]{
    			  createStructField("features",VectorType(), false),
    			  createStructField("label", DoubleType , false)
    			});

    	Dataset<Row> df1 = dataFrame.select("open","close");
    	List<Row> dd = df1.collectAsList();
    	List<Row> ll = new ArrayList<Row>();
    	
    	for(int i=0;i<dd.size();i++) {
    		if(!dd.get(i).anyNull())
    		ll.add( RowFactory.create(
    				 Vectors.sparse(1,new int[] {0},new double[] {Double.parseDouble((String)dd.get(i).getAs("open"))}),
    								  Double.parseDouble((String)dd.get(i).getAs("close"))
    								  )
    				);
    	}
    	Dataset<Row> dataset = spark.createDataFrame(ll, schema);
    	dataset.show(false);
    	dataset.printSchema();
    	/*
    	
    	RFormula formula = new RFormula()
    						.setFormula("close ~ open")
    						.setFeaturesCol("features")
    						.setLabelCol("label");
    	dataFrame = formula.fit(dataset).transform(dataset);
    	dataFrame.printSchema();
    	*/
    	dataFrame = dataset;
    	// on split les donnes en 2 partie 60 % pour l entraimenet et 40 % pour le teste
    	Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
    	Dataset<Row> train = splits[0];
    	Dataset<Row> test = splits[1];
    	
    	// specify layers for the neural network:
    	// input layer of size 1 (features), two intermediate of size 5 and 4
    	// and output of size 3 (classes)
    	 int[] layers = new int[] {1,64, 32, 16, 1000};
    	
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
    	System.out.println(evaluator);
    	
    
    }
}
