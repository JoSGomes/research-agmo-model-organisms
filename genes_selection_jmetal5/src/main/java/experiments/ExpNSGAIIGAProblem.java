/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import data_processing.ModelOrganism;
import data_processing.Preprocessor;
import general_algorithms.FileHandler;
import java.util.ArrayList;
import java.util.List;
import nsgaii.NSGAIIAlgorithm;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import weka.core.Instances;

/**
 *
 * @author pbexp
 */
public class ExpNSGAIIGAProblem {
    
    public static void main(String[] args) throws InterruptedException, ExecutionException, Exception, Exception {
        int populationSize = 100;
        int maxEvaluation = 20000;

        //Controle dos datasets
        int numberOfFolds = 10;
        int numberOfThreads = calculateNumThreads(numberOfFolds);

        Preprocessor preprocessor = null;
        String[] dataSets = {"BP", "MF", "CC", "BPMF", "BPCC", "MFCC", "BPMFCC"};
        String[] classifier = {"KNN", "NB", "J48"};

        System.out.println("Reading all the data...");
        HashMap<String, HashMap<String, HashMap<String, List<Instances>>>> allDatasets = FileHandler.readAllDatasetsFolds(dataSets);
        System.out.println("The read have been complete and the data are into memory!");

        System.out.println("The machine has " + Runtime.getRuntime().availableProcessors() + " cores processors");
        System.out.println("Using " + numberOfThreads + " threads for parallel execution.");

        for(String runningDataSet : dataSets){
            for(String runningClassifier : classifier){
                FileHandler.closeCSVWriter();
                String output = "results\\" + runningDataSet + runningClassifier + "\\results.csv";
                FileHandler.initCSVWriter(output);
                for(ModelOrganism organism : ModelOrganism.values()){
                    Map<Integer, Future<Object>> results = new HashMap<>();
                    ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
                    int indexThread = 1;
                    double probabilityCrossoverSelectInstances = 0;
                    double probabilityMutationSelectInstances = 0;
                    for(int fold = 0; fold < numberOfFolds; fold++){
                        //Parâmetros do NSGAII
                        switch (organism.originalDataset)
                        {
                            case "Drosophila melanogaster", "Saccharomyces cerevisiae" -> {
                                probabilityMutationSelectInstances = 0.2;
                                probabilityCrossoverSelectInstances = 0.5;
                            }
                            case "Mus musculus", "Caenorhabditis elegans" -> {
                                probabilityMutationSelectInstances = 0.2;
                                probabilityCrossoverSelectInstances = 0.9;
                            }
                        }
                        try
                        {
                            preprocessor = new Preprocessor(organism, runningDataSet, allDatasets, runningClassifier, fold, 1);

                            System.out.println("Starting... " + organism.originalDataset + " // " + runningDataSet+ " // FOLD - " + fold);
                            Callable<Object> experiment = new NSGAIIAlgorithm(
                                    preprocessor,
                                    populationSize,
                                    maxEvaluation,
                                    probabilityCrossoverSelectInstances,
                                    probabilityMutationSelectInstances,
                                    indexThread
                            );

                            Future<Object> submit = executor.submit(experiment);
                            results.put(indexThread, submit);
                            indexThread++;

                        }catch (Exception ex){
                            Logger.getLogger(NSGAIIAlgorithm.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }

                    //Espera da execução de cada thread.
                    for (int key : results.keySet()){
                        Future<Object> future = results.get(key);
                        Object aux = future.get();
                    }
                    executor.shutdown();
                }
            }
        }
    }
    
    // Calculates an adequate number of threads to process in parallel
    private static int calculateNumThreads(int numFolds) {
        int cores = Runtime.getRuntime().availableProcessors();
        
        int threads;       
        if (numFolds <= cores) { // process all folds at the same time
            threads = numFolds;
        } 
        else if (cores > numFolds / 2.0) { // balance the load in 2 batchs
            threads = (int) Math.ceil(numFolds / 2.0);
        } 
        else { // use all cores to process
            threads = cores;
        }
        return threads;

    } //end calculateNumThreads method

    private static int calculateNumThreads() {
        return Runtime.getRuntime().availableProcessors();
    }
}
