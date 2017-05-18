import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.nnet.learning.BackPropagation;

import java.util.Arrays;
import java.util.Random;


public class Network implements LearningEventListener {

    public static void main(String[] args) {

        new Network().run();

    }

    public void run(){
        // create training set
        DataSet trainingSet = new DataSet(30, 1);

        ArrfReader reader = new ArrfReader("/home/atomic/Desktop/IART/data/dataset.arff");

        for (int i = 0; i < reader.getFullDataSet().size(); i++) {
            trainingSet.addRow(new DataSetRow(reader.getPhishingData(i), reader.getPhishing(i)));
        }

// create multi layer perceptron
        MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 30, 2, 1);
// learn the training set


        LearningRule learningRule = myMlPerceptron.getLearningRule();

        //myMlPerceptron.getLearningRule().setMaxIterations(100);
        myMlPerceptron.getLearningRule().setLearningRate(0.7);

        learningRule.addListener(this);

        myMlPerceptron.learn(trainingSet);

// test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, trainingSet);

// save trained neural network
        myMlPerceptron.save("phishing_websites.nnet");

// load saved neural network
        NeuralNetwork loadedMlPerceptron = NeuralNetwork.createFromFile("phishing_websites.nnet");

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

    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError() + " : ");
        if (bp.getCurrentIteration()==1000)
            bp.stopLearning();
    }


}