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

    private ArrayList<ArrayList<Integer>> fullDataSet;

    public ArrfReader(String filePath) {
        this.filePath = filePath;
        this.fullDataSet = new ArrayList<>();

        this.readFile();
        this.readDataSet();
    }

    public ArrayList<Integer> readPhishingData(int instanceNum) {
        ArrayList<Integer> data = new ArrayList<>();

        List<String> dataString = Arrays.asList(this.dataInstances.instance(instanceNum).toString().split(","));

        for (int i = 0; i < dataString.size(); i++) {
            String s = dataString.get(i);

                data.add(Integer.parseInt(s));
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

    public static void main(String[] args) {

        ArrfReader reader = new ArrfReader("C:\\Users\\Vitor Esteves\\Documents\\IART\\data\\dataset.arff");

        System.out.println(reader.getPhishingData(0).get(1));
    }

    public ArrayList<Integer> getPhishingData(int number){
        ArrayList<Integer> ret = new ArrayList<>(this.fullDataSet.get(number));

        return ret;
    }

    private void readDataSet() {
        for (int i = 0; i < this.dataInstances.numInstances() - 1; i++) {
            this.fullDataSet.add(readPhishingData(i));
        }
    }
}