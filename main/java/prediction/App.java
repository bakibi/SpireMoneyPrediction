package prediction;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class App {

	public static void main(String[] args) {
		
		
		Logger.getLogger("org").setLevel(Level.OFF); // shut down log info.
		SparkSession spark = SparkSession.builder().master("local[*]").appName("DataProcess").getOrCreate();
		
		// load data from csv file
		Dataset<Row> data = spark.read().format("csv").option("header", true)
		        .load("stocks.csv")
		        .withColumn("openPrice", functions.col("open").cast("double")).drop("open")
		        .withColumn("lowPrice", functions.col("low").cast("double")).drop("low")
		        .withColumn("highPrice", functions.col("high").cast("double")).drop("high")
		        .withColumn("volumeTmp", functions.col("volume").cast("double")).drop("volume")
		        .withColumn("closePrice", functions.col("close").cast("double")).drop("close")
		        .toDF("date", "symbol", "open", "low", "high", "volume", "close");
		data.show(); // afficher les symboles 20
		
		Dataset<Row> symbols = data.select("date", "symbol").groupBy("symbol").agg(functions.count("date").as("count"));
		System.out.println(symbols.count());
		symbols.show();

	}

}
