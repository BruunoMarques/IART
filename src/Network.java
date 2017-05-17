import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


public class Network {
    public static void main(String[] args){
        // create training set (logical XOR function)
        DataSet trainingSet = new DataSet(30, 1);


        ArrfReader reader = new ArrfReader("/home/atomic/Desktop/IART/data/dataset.arff");

            for (int i = 0; i < reader.getFullDataSet().size(); i++) {
                trainingSet.addRow(new DataSetRow(reader.getPhishingData(i), reader.getPhishing(i)));
                System.out.print(reader.getPhishingData(i));
            }

// create multi layer perceptron
        MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.TANH, 30, 18, 1);
// learn the training set
        myMlPerceptron.learn(trainingSet);

// test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, trainingSet);

// save trained neural network
        myMlPerceptron.save("Phishing_Websites.nnet");

// load saved neural network
        NeuralNetwork loadedMlPerceptron = NeuralNetwork.createFromFile("Phishing_Websites.nnet");

// test loaded neural network
        System.out.println("Testing loaded neural network");
        testNeuralNetwork(loadedMlPerceptron, trainingSet);

    }

    public static void testNeuralNetwork(NeuralNetwork nnet, DataSet testSet) {

        for(DataSetRow dataRow : testSet.getRows()) {
            nnet.setInput(dataRow.getInput());
            nnet.calculate();
            double[ ] networkOutput = nnet.getOutput();
            System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
            System.out.println(" Output: " + Arrays.toString(networkOutput) );
        }

    }

}
