package weka;


import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

import java.io.FileReader;

public class Network {

    public static void simpleWekaTrain(String filepath) {
        try {
//Reading training arff or csv file
            FileReader trainreader = new FileReader(filepath);
            Instances train = new Instances(trainreader);
            train.setClassIndex(train.numAttributes() - 1);
//Instance of NN
            MultilayerPerceptron mlp = new MultilayerPerceptron();
//Setting Parameters
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(500);
            mlp.setHiddenLayers("a");
            mlp.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);
            System.out.println(eval.errorRate()); //Printing Training Mean root squared Error
            System.out.println(eval.toSummaryString()); //Summary of Training

        } catch (Exception ex) {
            ex.printStackTrace();
        }
        //L = Learning Rate
        //M = Momentum
        //N = Training Time or Epochs
        // H = Hidden Layers
    }





    public static void main(String[] args) {
        simpleWekaTrain("C:\\Users\\Vitor Esteves\\Documents\\IART\\data\\dataset.arff");
    }
}