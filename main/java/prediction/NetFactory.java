package prediction;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.mongodb.util.JSON;

import javafx.util.Pair;

public class NetFactory {
	private  MultiLayerNetwork myNet;
	private  StockDataSetIterator myIter;
	
	
	// pour initialiser le NN
	public   NetFactory() {
		myNet = RecurrentNets.buildLstmNetworks(5, 22);
		String file = "stocks.csv";
	    String symbol = "AAPL"; // stock name
	    int batchSize = 64; // mini-batch size
	    double splitRatio = 0.9; // 90% for training, 10% for testing
	    int epochs = 100; // training epochs
		myIter =  new StockDataSetIterator(file, symbol, batchSize, 22, splitRatio, PriceCategory.CLOSE);;
	}
	
	
	
	// 
	public  void trainNetwork(int epochs) {
		
		if(myNet == null || myIter == null)
			return ;
		 for (int i = 0; i < epochs; i++) {
	            while (myIter.hasNext()) myNet.fit(myIter.next()); // fit model using mini-batch data
	            myIter.reset(); // reset iterator
	            myNet.rnnClearPreviousState(); // clear previous state
	        }
	}
	
	
	
	public  void saveNetworkModel(String path) {
		File locationToSave = new File(path+"/modelNetwork.zip");
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        try {
			ModelSerializer.writeModel(myNet, locationToSave, true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
	
	public  int loadModelFromFile(String fichier) {
		File f = new File(fichier);
		if(f.exists()) {
			try {
				myNet = ModelSerializer.restoreMultiLayerNetwork(f);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return 1;
		}
		return 0;
	}
	
	
	
	public  double predictOneSeq(List<Point> list,PriceCategory category,int maxGetting,int MaxPredic) {
		
		double ans = 0;

		//Pair<INDArray, INDArray> test = myIter.generateTestDataSetOne(list,maxGetting, MaxPredic);
		Pair<INDArray, INDArray> test = myIter.getTestDataSet().get(0);
		double max = myIter.getMaxNum(category);
        double min = myIter.getMinNum(category);
        ans =  myNet.rnnTimeStep(test.getKey()).getDouble(maxGetting - 1) * (max - min) + min;
		
		return ans;
	}
	
	
	
	
	public MultiLayerNetwork getMyNet() {
		return myNet;
	}



	public void setMyNet(MultiLayerNetwork myNet) {
		this.myNet = myNet;
	}



	public StockDataSetIterator getMyIter() {
		return myIter;
	}



	public void setMyIter(StockDataSetIterator myIter) {
		this.myIter = myIter;
	}



	public List<Pair<String,Double>> predictPreriodTime(List<Point> list,PriceCategory category,int maxGetting,int Time) {
		List<Pair<String,Double>> ans= new ArrayList<Pair<String,Double>>();
		Vector<String> dates = next(Time);
		double nv_val = 0;
		for(int i=0;i<Time;i++) {
			nv_val = predictOneSeq(list, category, maxGetting, 1);
			ans.add(new Pair<String,Double>(dates.get(i),nv_val));
			Point p = list.get(0);
			list.remove(0);
			p.setClose(nv_val);
			list.add(list.size(), p);
		}
		
		return ans;
	}
	
	
	
	public JSONObject predictObject(List<Point> list,PriceCategory category,String symb) {
		JSONObject obj = new JSONObject();
		obj.put("symbol", symb);
		JSONArray week_arr = new JSONArray();
		JSONArray mois_arr = new JSONArray();
		JSONArray trimestre_arr = new JSONArray();
		JSONArray year_arr = new JSONArray();
		JSONArray max_arr = new JSONArray();
		
		int week_size = 5;
		int mois_size = week_size*4;
		int trimestre_size = mois_size*3;
		int year_size = mois_size*12;
		int max_size = 5*year_size;
		List<Pair<String,Double>> ls = predictPreriodTime(list, category, 22,max_size );
		for(int i=0;i<week_size;i++) {
			week_arr.put(new JSONObject().put("prediction", ls.get(i).getValue()).put("date", ls.get(i).getKey()));
		}
		for(int i=0;i<mois_size;i++) {
			mois_arr.put(new JSONObject().put("prediction", ls.get(i).getValue()).put("date", ls.get(i).getKey()));
		}
		
		for(int i=0;i<trimestre_size;i++) {
			trimestre_arr.put(new JSONObject().put("prediction", ls.get(i).getValue()).put("date", ls.get(i).getKey()));
		}
		for(int i=0;i<year_size;i++) {
			year_arr.put(new JSONObject().put("prediction", ls.get(i).getValue()).put("date", ls.get(i).getKey()));
		}
		for(int i=0;i<max_size;i++) {
			max_arr.put(new JSONObject().put("prediction", ls.get(i).getValue()).put("date", ls.get(i).getKey()));
		}
		
		obj.put("week",week_arr);
		obj.put("month",mois_arr);
		obj.put("trimestre",trimestre_arr);
		obj.put("year",year_arr);
		obj.put("max", max_arr);
		return obj;
	}
	public static boolean isHoliday(Date d) {
		String hol[] = {"01-01","01-05","02-19","03-30","05-28","07-04","09-03","11-22","12-25"};
		SimpleDateFormat resFormat = new SimpleDateFormat("MM-dd");
		String strcmp = resFormat.format(d);
		for(String str:hol) {
			if(str.compareTo(strcmp) == 0)
				return true;
				
		}
		return false;
	}
	public static Vector<String> next(int nbr) {
		 Date now = new Date();
		 Vector<String> ans = new Vector<String>();
	     SimpleDateFormat simpleDateformat = new SimpleDateFormat("E"); // the day of the week abbreviated
	     SimpleDateFormat resFormat = new SimpleDateFormat("YYYY-MM-dd HH:mm:00");
	     System.out.println(simpleDateformat.format(now));
	     
	     Calendar calendar = Calendar.getInstance();
	     calendar.setTime(now);
	     calendar.add(Calendar.DATE,1);
	        System.out.println(calendar.get(Calendar.DAY_OF_WEEK)); // the day of the week in numerical format
		for(int i=0;i<nbr;i++) {
			 calendar.add(Calendar.DATE,1);
			 if(calendar.get(Calendar.DAY_OF_WEEK) ==calendar.SATURDAY ||
					calendar.get(Calendar.DAY_OF_WEEK) ==calendar.SUNDAY || isHoliday(calendar.getTime())){
				 i--;
			 }
			 else {
				 ans.add(resFormat.format(calendar.getTime())); // the day of the week in numerical format
			 }
		}
		
		return ans;
	}
	
	

}
