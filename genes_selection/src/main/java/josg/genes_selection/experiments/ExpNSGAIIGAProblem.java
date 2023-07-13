/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package josg.genes_selection.experiments;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import josg.genes_selection.data_processor.ModelOrganism;
import josg.genes_selection.data_processor.Preprocessor;
import josg.genes_selection.nsgaii.NSGAIIAlgorithm;

/**
 *
 * @author pbexp
 */
public class ExpNSGAIIGAProblem {
    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        
        int populationSize = 100;
        int maxEvaluation = 20000;
        //Execução paralela
        int numberOfFolds = 10;
        int numberOfThreads;  
        
        //Controle dos datasets
        Preprocessor preprocessor;
        numberOfThreads = calculateNumThreads(numberOfFolds);

        System.out.println("The machine has " + Runtime.getRuntime().availableProcessors() + " cores processors");
        System.out.println("Using " + numberOfThreads + " threads for parallel execution.");
        
        Map<Integer, Future<Object>> results = new HashMap<>();
        for(ModelOrganism organism : ModelOrganism.values()){            
            ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
            int indexThread = 1;
            
            for(int n = 0; n < numberOfFolds; n++)
            {
                //Parâmetros do NSGAII
                int maxEvaluationsSelectInstances = maxEvaluation;         
                int populationSizeSelectInstances = populationSize;
                double probabilityCrossoverSelectInstances = 0;
                double probabilityMutationSelectInstances = 0;
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
                    preprocessor = new Preprocessor(organism, n);

                    System.out.println("\n" + organism.originalDataset);                                    
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

                }catch (Exception ex) 
                {
                    Logger.getLogger(NSGAIIAlgorithm.class.getName()).log(Level.SEVERE, null, ex);
                }                   
                       
            }                  
            //laço dedicado a espera da execução da thread.           
            for (int key : results.keySet()) {
            Future<Object> future = results.get(key);
            Object aux = future.get();                
            }            
            executor.shutdown(); 
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
