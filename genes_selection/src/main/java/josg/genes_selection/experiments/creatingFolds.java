/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package josg.genes_selection.experiments;

import java.io.FileWriter;
import java.util.Random;
import josg.genes_selection.data_processor.ModelOrganism;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author pbexp
 */
public class creatingFolds {
    public static void main(String[] args) throws Exception {
        int numFolds = 10;
        /*
        
        double totalSen = 0;
        double totalSpe = 0;
        
        double GMean;
        
        double sumGMean = 0;
        double[] GMeans = new double[10];
        double[] sensivity = new double[10]; 
        double[] specificity = new double[10]; 
        
        sensivity[0] = 0.06923076923076923;
        sensivity[1] = 0.075;
        sensivity[2] = 0.1;
        sensivity[3] = 0.0642857142857143;
        sensivity[4] = 0.045454545454545456;
        sensivity[5] = 0.058333333333333334;
        sensivity[6] = 0.07272727272727272;
        sensivity[7] = 0.08888888888888888;
        sensivity[8] = 0.08333333333333334;
        sensivity[9] = 0.07272727272727272;
        
        specificity[0] = 0.08;
        specificity[1] = 0.06666666666666667;
        specificity[2] = 0.025;
        specificity[3] = 0.05;
        specificity[4] = 0.05714285714285714;
        specificity[5] = 0.06666666666666667;
        specificity[6] = 0.02857142857142857;
        specificity[7] = 0.05555555555555556;
        specificity[8] = 0.0;
        specificity[9] = 0.08333333333333334;
        
        for(int i = 0; i < 10; i++){
            GMeans[i] = (Math.round((Math.sqrt( (sensivity[i] * 10) * (specificity[i] * 10)))*10000.00)/10000.00)*100.00;
            totalSpe += sensivity[i];
            totalSen += specificity[i];
            sumGMean += GMeans[i]; 
        }
        
        
        
        GMean = (Math.round((Math.sqrt(totalSen * totalSpe))*10000.00)/10000.00)*100.00;
        
        System.out.println("GMean das médias: " + GMean);
        System.out.println("Média das GMeans: "+ sumGMean/10);*/
        FileWriter writer;
        int[] thresholds = {3,4,5,6,7,8,9,10};
        
        for (int threshold : thresholds)
        {
            
            for(ModelOrganism organism : ModelOrganism.values()){
                
                for(int k=0;k < numFolds;k++)
                {
                    String file = "src\\main\\datasets\\threshold-" + threshold + "\\"+ organism.originalDataset +"-dataset-folds\\" + organism.originalDataset + "TRA-" + k +"-fold.arff"; //+ organism.originalDataset +"TRA-0-fold.arff";
                    DataSource source = new DataSource(file);

                    Instances dataset = source.getDataSet();       


                    if(dataset.classIndex() == -1 ){
                            dataset.setClassIndex(dataset.numAttributes() -1);           
                    }
                    Random random = new Random(1);
                    dataset.stratify(numFolds);
                    Instances[] split = new Instances[2];
                    
                    split[0] = dataset.trainCV(numFolds, 0);
                    split[1] = dataset.testCV(numFolds, 0);

                    split[0].setRelationName(organism.originalDataset + "-threshold"+ threshold + "-TRA-"+ k + "-fold");
                    split[1].setRelationName(organism.originalDataset + "-threshold"+ threshold + "-TEST-"+ k + "-fold");
                    split[0].randomize(random);
                    split[1].randomize(random);
                    
                    String String_TRA = split[0].toString();
                    String String_TEST = split[1].toString();

                    writer = new FileWriter("src\\main\\datasets\\threshold-"+ threshold +"\\" + organism.originalDataset + "-dataset-folds\\test-train-agmo\\" 
                                            + organism.originalDataset+ "TRA-" + k +"-fold.arff");
                    writer.write(String_TRA);
                    writer.flush();
                    writer = new FileWriter("src\\main\\datasets\\threshold-" + threshold +"\\" + organism.originalDataset + "-dataset-folds\\test-train-agmo\\" 
                                            + organism.originalDataset+ "TEST-" + k +"-fold.arff");               
                    writer.write(String_TEST);
                    writer.flush();
                }

            }
        }
        
         
        /*List<String[]> resultsCSV = new ArrayList<String[]>();
        String[] headers = new String[]{"mutationProbability", "crossoverProbability", "selectionProbability", "GMean" , "ratioReduction"};
        String[] test = {"test", "test"};
        String[] test2 = {"test2", "test2"};
        String[] test3 = {"test3", "test3"};
        resultsCSV.add(headers);
        resultsCSV.add(test);
        resultsCSV.add(test2);
        resultsCSV.add(test3);
        

        String outputFile = "results\\results.csv";
        
        Writer writer = new FileWriter(outputFile);
        CSVWriter csvWriter = new CSVWriter(writer);
        csvWriter.writeAll(resultsCSV);
        csvWriter.close();
        writer.close();*/
    }
} 

