package neuroph;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class ArrfReader {

    private String filePath;
    private Instances dataInstances;

    private ArrayList<ArrayList<Double>> fullDataSet;

    public ArrayList<ArrayList<Double>> getFullDataSet() {
        return fullDataSet;
    }

    public ArrfReader(String filePath) {
        this.filePath = filePath;
        this.fullDataSet = new ArrayList<>();

        this.readFile();
        this.readDataSet();
    }

    public ArrayList<Double> readPhishingData(int instanceNum) {
        ArrayList<Double> data = new ArrayList<>();

        List<String> dataString = Arrays.asList(this.dataInstances.instance(instanceNum).toString().split(","));

        for (int i = 0; i < dataString.size(); i++) {
            String s = dataString.get(i);

                data.add(Double.parseDouble(s));

        }

        return data;
    }

    private void readFile() {

        /*
        Reads data from an ARFF file, either in incremental or batch mode.
        Typical code for batch usage:
        */

        try {
            BufferedReader reader = new BufferedReader(new FileReader(this.filePath));
            this.dataInstances = new Instances(reader);
            reader.close();
            dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Double> getPhishing(int websiteNum)
    {
        ArrayList<Double> ret = new ArrayList<>();
        double phishing = this.fullDataSet.get(websiteNum).get(30);

        ret.add(phishing);

        return ret;
    }


    public ArrayList<Double> getPhishingData(int number){
        ArrayList<Double> ret = new ArrayList<>(this.fullDataSet.get(number));
        ret.remove(30);
        return ret;

    }

    private void readDataSet() {
        for (int i = 0; i < this.dataInstances.numInstances() - 1; i++) {
            this.fullDataSet.add(readPhishingData(i));
        }
    }

    public static void main(String[] args) {

        ArrfReader reader = new ArrfReader("/home/atomic/Desktop/IART/data/dataset.arff");

        System.out.println(reader.getPhishingData(1).get(1));
    }
}