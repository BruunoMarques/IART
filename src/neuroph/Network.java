package neuroph;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.sample.SubSampling;

import java.util.Arrays;
import java.util.List;


public class Network implements LearningEventListener {

    public static void main(String[] args) {

        new Network().run();

    }

    public void run(){
        // create training set
        DataSet dataSet = new DataSet(30, 1);

        ArrfReader reader = new ArrfReader("C:\\Users\\Vitor Esteves\\Documents\\IART\\data\\dataset.arff");

        for (int i = 0; i < reader.getFullDataSet().size(); i++) {
            dataSet.addRow(new DataSetRow(reader.getPhishingData(i), reader.getPhishing(i)));
        }

// create multi layer perceptron
        MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 30, 21, 1);
// learn the training set

        MomentumBackpropagation bp = new MomentumBackpropagation();
        myMlPerceptron.setLearningRule(bp);

        LearningRule learningRule = myMlPerceptron.getLearningRule();

        bp.setNeuralNetwork(myMlPerceptron);
        bp.setMomentum(0.7);
        bp.setLearningRate(0.2);

        learningRule.addListener(this);

        SubSampling subSampling = new SubSampling(70,30);
        List<DataSet> dataSetList = subSampling.sample(dataSet);

        DataSet trainingDataset = dataSetList.get(0);
        DataSet testingDataset = dataSetList.get(1);


        myMlPerceptron.learn(trainingDataset, bp);

// test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, testingDataset);

// save trained neural network
        myMlPerceptron.save("phishing_websites.nnet");

// load saved neural network
        NeuralNetwork loadedMlPerceptron = NeuralNetwork.createFromFile("phishing_websites.nnet");

// test loaded neural network
        System.out.println("Testing loaded neural network");
        testNeuralNetwork(loadedMlPerceptron, testingDataset);

    }

    public static void testNeuralNetwork(NeuralNetwork nnet, DataSet testSet) {
        double test_size = 0;
        double correctCounter = 0;
        double accuracy = 0;
        for(DataSetRow dataRow : testSet.getRows()) {
            nnet.setInput(dataRow.getInput());
            nnet.calculate();
            double[] networkOutput = nnet.getOutput();
            double desiredOutput = Double.parseDouble(Arrays.toString(dataRow.getDesiredOutput()).replace("[", "").replace("]", ""));
            double actualOutput = Double.parseDouble(Arrays.toString(networkOutput).replace("[", "").replace("]", ""));
            double diff = Math.abs(actualOutput - desiredOutput);
            test_size+=1;
            if(Math.abs(networkOutput[0] - dataRow.getDesiredOutput()[0]) < 0.05){
                correctCounter += 1;
            }
            System.out.println("\tDesired output: " + desiredOutput + " \t|\t Actual output: " + actualOutput + " \t|\t Difference: " + diff);
        }
        accuracy = correctCounter / test_size;
        System.out.println("\tAccuracy: " + accuracy);


    }


    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation bp = (MomentumBackpropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError() + " : ");
        if (bp.getCurrentIteration()==10)
            bp.stopLearning();
    }


}