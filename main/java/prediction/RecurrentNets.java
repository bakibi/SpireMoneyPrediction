package prediction;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
/**
 * 
 * @author ELKAISSI SOUHAIL
 *
 */


public class RecurrentNets {

	// le taux d'apprentissage il determine la taille d'un etape , quand on change le poid
	private static final double learningRate = 0.05;
	// un changement d'un model du NN 
	private static final int iterations = 1;
	// pour la configuration du poid du reseau
	private static final int seed = 12345;
	// 
    private static final int lstmLayer1Size = 256;
    private static final int lstmLayer2Size = 256;
    private static final int denseLayerSize = 32;
    private static final double dropoutRatio = 0.2;
    private static final int truncatedBPTTLength = 22;


    
		public static MultiLayerNetwork buildLstmNetworks(int nIn, int nOut) {
			Logger.getLogger("org").setLevel(Level.OFF); // shut down log info.
			SparkSession spark = SparkSession.builder().master("local[*]").appName("DataProcess").getOrCreate();
			SparkContext sc = spark.sparkContext();
			
		
			
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)//poid
                .iterations(iterations)//changement de modele
                .learningRate(learningRate)// taux d'apprentissage
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)//
                .weightInit(WeightInit.XAVIER)//  randomly
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l2(1e-4)// regularisation  
                .list()
                /*
                 * configuration de la premiere couche "input"
                 *  Cette couche est consitue d 'nIn' noeud entant 
                 *  et de 'lstmLayer1Size' noeud Sortant
                 *  Pour la fonction d'activation j'utilise le sigmoid 
                 * */
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn)
                        .nOut(lstmLayer1Size)
                        .activation(Activation.TANH) // la fonction tang
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                //configuratgion de la deuxieme couche 
                .layer(1, new GravesLSTM.Builder()
                        .nIn(lstmLayer1Size)
                        .nOut(lstmLayer2Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                //  normale 
                .layer(2, new DenseLayer.Builder()
                		.nIn(lstmLayer2Size)
                		.nOut(denseLayerSize)
                		.activation(Activation.RELU) // non linaier activation
                		.build())
                .layer(3, new RnnOutputLayer.Builder()
                        .nIn(denseLayerSize)
                        .nOut(nOut)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(truncatedBPTTLength)
                .tBPTTBackwardLength(truncatedBPTTLength)
                .pretrain(false)
                .backprop(true)
                .build();

        //	instancier le reseaux 
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        // initialiser le reseaux
        net.init();
        // pour stocker le score du modele en interne
        net.setListeners(new ScoreIterationListener(100));
    	VoidConfiguration voidConfiguration = VoidConfiguration.builder()
	            .unicastPort(40123)
	            .build();
    	
    	
    	int examplesPerDataSetObject = 1;
    	TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
    	        .build();
		SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf);
		
		
        return net;
    }
}
