/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;


import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
        //Execução paralela
        int numberOfFolds = 10;
        int numberOfThreads;  
        
        //Controle dos datasets
        Preprocessor preprocessor = null;
        numberOfThreads = calculateNumThreads(numberOfFolds);
        String[] dataSet = {"", "threshold-3\\", "threshold-4\\", "threshold-5\\", "threshold-6\\", "threshold-7\\", "threshold-8\\"};
        String[] classifier = {"KNN", "NB", "J48"};
        System.out.println("The machine has " + Runtime.getRuntime().availableProcessors() + " cores processors");
        System.out.println("Using " + numberOfThreads + " threads for parallel execution.");
        
        int maxEvaluationsSelectInstances = maxEvaluation;
        int populationSizeSelectInstances = populationSize;
        for(String runningDataSet : dataSet){    
            for(String runningClassifier : classifier){
                for(ModelOrganism organism : ModelOrganism.values()){
                    double probabilityCrossoverSelectInstances = 0;
                    double probabilityMutationSelectInstances = 0;
                    for(int n = 0; n < numberOfFolds; n++){
                        Map<Integer, Future<Object>> results = new HashMap<>();
                        ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
                        int indexThread = 1;
                        for(int k = 0; k < 3; k++){
                            //Parâmetros do NSGAII                         
                            switch (organism.originalDataset) 
                            {
                                case "fly" -> {
                                    probabilityMutationSelectInstances = 0.2;
                                    probabilityCrossoverSelectInstances = 0.5;
                                }
                                case "mouse" -> {
                                    probabilityMutationSelectInstances = 0.2;
                                    probabilityCrossoverSelectInstances = 0.9;
                                }
                                case "worm" -> {
                                    probabilityMutationSelectInstances = 0.2;
                                    probabilityCrossoverSelectInstances = 0.9;
                                }
                                case "yeast" -> {
                                    probabilityMutationSelectInstances = 0.2;
                                    probabilityCrossoverSelectInstances = 0.5;
                                }
                            }
                            try 
                            {
                                preprocessor = new Preprocessor(organism, n, runningDataSet, runningClassifier);

                                System.out.println(organism.originalDataset + " fold-" + n + " " + runningDataSet);
                                Callable<Object> experiment = new NSGAIIAlgorithm(
                                    preprocessor, 
                                    populationSizeSelectInstances,
                                    maxEvaluationsSelectInstances,
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
                        //laço dedicado a espera da execução da thread.           
                        for (int key : results.keySet()){
                        Future<Object> future = results.get(key);
                        Object aux = future.get();                
                        }
                        
                        //Testar para a melhor solução gerada    
                        int bestSolutionCounter = 0;
                        double bestGMean = 0, bestRatioR = 0, auxGMean, auxRatioR;
                        ArrayList solAndPopulation;
                        BinarySolution auxSolution;        
                        for(int i = 1; i < (results.size() + 1); i++){
                            solAndPopulation = (ArrayList) results.get(i).get();
                            auxSolution =  (BinarySolution) solAndPopulation.get(0);
                            auxGMean = (double) auxSolution.objectives()[0]*(-1);
                            auxRatioR = (double) auxSolution.objectives()[1]*(-1);
                            if(auxGMean > bestGMean || (auxGMean == bestGMean && auxRatioR > bestRatioR)){
                                bestSolutionCounter = i;
                                bestGMean = auxGMean;
                                bestRatioR = auxRatioR;
                            }         
                        }
                        
                        ArrayList<Instances> trainFold = preprocessor.getDatasetsTRAFolds();                  
                        ArrayList<Instances> testFold = preprocessor.getDatasetsTESTFolds();

                        solAndPopulation = (ArrayList) results.get(bestSolutionCounter).get();
                        List<BinarySolution> population = (List<BinarySolution>) solAndPopulation.get(1); //Para o VAR e FUN

                        double[] resultsClassify = preprocessor.getClassifier().classifySolution( (BinarySolution) solAndPopulation.get(0), trainFold, testFold);
                        SolutionListOutput solListOutput = new SolutionListOutput(population);
                        solListOutput
                                    .setVarFileOutputContext(new DefaultFileOutputContext("results\\" + runningDataSet + runningClassifier + "\\" + "VAR-" + preprocessor.getOrganism().originalDataset + "\\fold-" + preprocessor.getFold() + ".csv", ","))
                                    .setFunFileOutputContext(new DefaultFileOutputContext("results\\" + runningDataSet + runningClassifier + "\\" + "FUN-" + preprocessor.getOrganism().originalDataset + "\\fold-" + preprocessor.getFold() + ".csv", ","))
                                    .print();
                        String output = "results\\" + runningDataSet + runningClassifier + "\\results.csv";
                        FileHandler.saveResults(output, populationSize, probabilityMutationSelectInstances, probabilityCrossoverSelectInstances, resultsClassify, preprocessor.getOrganism().originalDataset, preprocessor.getFold(), bestGMean);  
                        
                        executor.shutdown();  
                  
                    }                  
                }
            }
        }
        
            
    }
    
    // Calculates a adequate number of threads to process in parallel
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
        int cores = Runtime.getRuntime().availableProcessors();
        return cores;
    }
}
