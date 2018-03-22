package prediction;

import com.google.common.collect.ImmutableMap;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * 
 * @author ELKAISSI SOUHAIL
 *
 */

public class StockDataSetIterator implements DataSetIterator {

    // category and its index 
    private final Map<PriceCategory, Integer> featureMapIndex = ImmutableMap.of(
    		PriceCategory.OPEN, 0,
    		PriceCategory.CLOSE, 1,
            PriceCategory.LOW, 2,
            PriceCategory.HIGH, 3,
            PriceCategory.VOLUME, 4
            );
    // number of features for a stock data
    private final int VECTOR_SIZE = 5;
    // mini-batch size
    private int miniBatchSize;
    // default 22 jours
    private int exampleLength = 22; 
    // 	default 1 la taille des donnees à predire
    private int predictLength = 1; 
    // minimal values of each feature in stock dataset
    private double[] minArray = new double[VECTOR_SIZE];
    // maximal values of each feature in stock dataset 
    private double[] maxArray = new double[VECTOR_SIZE];
    // feature to be selected as a training target
    private PriceCategory category;
    // mini-batch offset 
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<Integer>();
    // stock dataset for training 
    private List<Point> train;
    // adjusted stock dataset for testing 
    private List<Pair<INDArray, INDArray>> test;
    
    
    public StockDataSetIterator (String filename, String symbol, int miniBatchSize, int exampleLength, double splitRatio, PriceCategory category) {
        List<Point> stockDataList = readStockDataFromFile(filename, symbol);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(stockDataList.size() * splitRatio);
        train = stockDataList.subList(0, split);
        test = generateTestDataSet(stockDataList.subList(split, stockDataList.size()));
        initializeOffsets();
    }

    /** 
     * initialize the mini-batch offsets
     * 
     */
    private void initializeOffsets () {
        exampleStartOffsets.clear();
        int window = exampleLength + predictLength;
        for (int i = 0; i < train.size() - window; i++) { exampleStartOffsets.add(i); }
    }
    

    public List<Pair<INDArray, INDArray>> getTestDataSet() { return test; }
    public double[] getMaxArray() { return maxArray; }
    public double[] getMinArray() { return minArray; }
    public double getMaxNum (PriceCategory category) { return maxArray[featureMapIndex.get(category)]; }
    public double getMinNum (PriceCategory category) { return minArray[featureMapIndex.get(category)]; }

    
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;
        if (category.equals(PriceCategory.ALL)) label = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        else label = Nd4j.create(new int[] {actualMiniBatchSize, predictLength, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            Point curData = train.get(startIdx);
            Point nextData;
            for (int i = startIdx; i < endIdx; i++) {
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, (curData.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
                input.putScalar(new int[] {index, 1, c}, (curData.getClose() - minArray[1]) / (maxArray[1] - minArray[1]));
                input.putScalar(new int[] {index, 2, c}, (curData.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                input.putScalar(new int[] {index, 3, c}, (curData.getHigh() - minArray[3]) / (maxArray[3] - minArray[3]));
                input.putScalar(new int[] {index, 4, c}, (curData.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
                nextData = train.get(i + 1);
                if (category.equals(PriceCategory.ALL)) {
                    label.putScalar(new int[] {index, 0, c}, (nextData.getOpen() - minArray[1]) / (maxArray[1] - minArray[1]));
                    label.putScalar(new int[] {index, 1, c}, (nextData.getClose() - minArray[1]) / (maxArray[1] - minArray[1]));
                    label.putScalar(new int[] {index, 2, c}, (nextData.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                    label.putScalar(new int[] {index, 3, c}, (nextData.getHigh() - minArray[3]) / (maxArray[3] - minArray[3]));
                    label.putScalar(new int[] {index, 4, c}, (nextData.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
                } else {
                    label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(Point data) {
        double value;
        switch (category) {
            case OPEN: value = (data.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]); break;
            case CLOSE: value = (data.getClose() - minArray[1]) / (maxArray[1] - minArray[1]); break;
            case LOW: value = (data.getLow() - minArray[2]) / (maxArray[2] - minArray[2]); break;
            case HIGH: value = (data.getHigh() - minArray[3]) / (maxArray[3] - minArray[3]); break;
            case VOLUME: value = (data.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]); break;
            default: throw new NoSuchElementException();
        }
        return value;
    }

     public int totalExamples() { return train.size() - exampleLength - predictLength; }

    public int inputColumns() { return VECTOR_SIZE; }

   public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return VECTOR_SIZE;
        else return predictLength;
    }
   
   
    public boolean resetSupported() { return false; }
    public boolean asyncSupported() { return false; }
     
    public void reset() { initializeOffsets(); }
    
    public int batch() { return miniBatchSize; }
   
    
    public int cursor() { return totalExamples() - exampleStartOffsets.size(); }

     public int numExamples() { return totalExamples(); }

    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

     public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

     public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

     public boolean hasNext() { return exampleStartOffsets.size() > 0; }

     public DataSet next() { return next(miniBatchSize); }
    /**
     *  Generer le dataset from un ensemble de point
     * @param stockDataList
     * @return
     */
    public List<Pair<INDArray, INDArray>> generateTestDataSet (List<Point> stockDataList) {
    	int window = exampleLength + predictLength;
    	List<Pair<INDArray, INDArray>> test = new ArrayList<Pair<INDArray, INDArray>>();
    	for (int i = 0; i < stockDataList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f'); // f pour la construction rapide (fast)
    		for (int j = i; j < i + exampleLength; j++) {
    			Point stock = stockDataList.get(j);
    			input.putScalar(new int[] {j - i, 0}, (stock.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
    			input.putScalar(new int[] {j - i, 1}, (stock.getClose() - minArray[1]) / (maxArray[1] - minArray[1]));
    			input.putScalar(new int[] {j - i, 2}, (stock.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
    			input.putScalar(new int[] {j - i, 3}, (stock.getHigh() - minArray[3]) / (maxArray[3] - minArray[3]));
    			input.putScalar(new int[] {j - i, 4}, (stock.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
    		}
            Point stock = stockDataList.get(i + exampleLength);
            INDArray label;
            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');  // f pour la construction rapide (fast)
                label.putScalar(new int[] {0}, stock.getOpen());
                label.putScalar(new int[] {1}, stock.getClose());
                label.putScalar(new int[] {2}, stock.getLow());
                label.putScalar(new int[] {3}, stock.getHigh());
                label.putScalar(new int[] {4}, stock.getVolume());
            } else {
                label = Nd4j.create(new int[] {1}, 'f');
                switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
                    default: throw new NoSuchElementException();
                }
            }
    		test.add(new Pair<INDArray, INDArray>(input, label));
    	}
    	return test;
    }
    
    public Pair<INDArray, INDArray> generateTestDataSetOne (List<Point> stockDataList,int exampleLength,int predictLength) {
    	Pair<INDArray, INDArray> ans ;
    	
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f'); // f pour la construction rapide (fast)
    		for (int j = 0; j < exampleLength-1; j++) {
    			Point stock = stockDataList.get(j);
    			input.putScalar(new int[] {j , 0}, (stock.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
    			input.putScalar(new int[] {j , 1}, (stock.getClose() - minArray[1]) / (maxArray[1] - minArray[1]));
    			input.putScalar(new int[] {j , 2}, (stock.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
    			input.putScalar(new int[] {j , 3}, (stock.getHigh() - minArray[3]) / (maxArray[3] - minArray[3]));
    			input.putScalar(new int[] {j , 4}, (stock.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
    		}
            Point stock = stockDataList.get(exampleLength-1);
            INDArray label;
            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');  // f pour la construction rapide (fast)
                label.putScalar(new int[] {0}, stock.getOpen());
                label.putScalar(new int[] {1}, stock.getClose());
                label.putScalar(new int[] {2}, stock.getLow());
                label.putScalar(new int[] {3}, stock.getHigh());
                label.putScalar(new int[] {4}, stock.getVolume());
            } else {
                label = Nd4j.create(new int[] {1}, 'f');
                switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
                    default: throw new NoSuchElementException();
                }
            }
    		ans = new Pair<INDArray, INDArray>(input, label);
    	
    	return ans;
    }

	public List<Point> readStockDataFromFile (String filename, String symbol) {
        List<Point> stockDataList = new ArrayList<Point>();
        
        try {
            for (int i = 0; i < maxArray.length; i++) { // initialize max and min arrays
                maxArray[i] = Double.MIN_VALUE;
                minArray[i] = Double.MAX_VALUE;
            }
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll(); // load all elements in a list
            System.out.println("size of the element read "+list.size());
            for (String[] arr : list) {
                if (!arr[1].equals(symbol)) continue;
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length - 2; i++) {
                    nums[i] = Double.valueOf(arr[i + 2]);
                    if (nums[i] > maxArray[i]) maxArray[i] = nums[i];
                    if (nums[i] < minArray[i]) minArray[i] = nums[i];
                }
                	
                stockDataList.add(new Point(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3], nums[4]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stockDataList;
    }

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public void setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
	}

	public int getExampleLength() {
		return exampleLength;
	}

	public void setExampleLength(int exampleLength) {
		this.exampleLength = exampleLength;
	}

	public int getPredictLength() {
		return predictLength;
	}

	public void setPredictLength(int predictLength) {
		this.predictLength = predictLength;
	}

	public PriceCategory getCategory() {
		return category;
	}

	public void setCategory(PriceCategory category) {
		this.category = category;
	}

	public LinkedList<Integer> getExampleStartOffsets() {
		return exampleStartOffsets;
	}

	public void setExampleStartOffsets(LinkedList<Integer> exampleStartOffsets) {
		this.exampleStartOffsets = exampleStartOffsets;
	}

	public List<Point> getTrain() {
		return train;
	}

	public void setTrain(List<Point> train) {
		this.train = train;
	}

	public List<Pair<INDArray, INDArray>> getTest() {
		return test;
	}

	public void setTest(List<Pair<INDArray, INDArray>> test) {
		this.test = test;
	}

	public Map<PriceCategory, Integer> getFeatureMapIndex() {
		return featureMapIndex;
	}

	public int getVECTOR_SIZE() {
		return VECTOR_SIZE;
	}

	public void setMinArray(double[] minArray) {
		this.minArray = minArray;
	}

	public void setMaxArray(double[] maxArray) {
		this.maxArray = maxArray;
	}
}
