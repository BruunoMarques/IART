import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import weka.core.Instances;
import java.util.ArrayList;


public class ArrfReader {

    private String filePath;
    private Instances dataInstances;
    private ArrayList<ArrayList<Double>> fullDataSet;

    public ArrfReader(String filePath)
    {
        this.filePath = filePath;
        this.fullDataSet = new ArrayList<>();

        this.readFile();
        //this.readDataSet();
    }


    private void readFile(){

        try{
        BufferedReader reader=new BufferedReader(new FileReader(this.filePath));
        this.dataInstances=new Instances(reader);
        reader.close();

        // setting class attribute
        dataInstances.setClassIndex(dataInstances.numAttributes()-1);
        }
                catch(FileNotFoundException e)
                {
                    e.printStackTrace();
                }
                catch(IOException e)
                {
                    e.printStackTrace();
                }
        }
    public static void main(String[] args) {

        ArrfReader reader = new ArrfReader("");

        System.out.print(reader);
    }
}