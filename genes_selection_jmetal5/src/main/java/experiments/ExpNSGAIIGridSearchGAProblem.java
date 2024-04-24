package experiments;

import data_processing.ModelOrganism;
import data_processing.Preprocessor;
import general_algorithms.FileHandler;
import nsgaii.NSGAIIAlgorithm;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ExpNSGAIIGridSearchGAProblem {

    public static void main(String[] args) throws Exception {
        int maxEvaluation = 20000;

        // Grid
        float[] mutationSearch = {0.2F, 0.4F, 0.6F};
        float[] crossoverSearch = {0.5F, 0.7F, 0.9F};
        int[] populationSearch = {100, 150, 200};
        int[] kValueSearch = {1, 5, 10};

        // Controle dos datasets
        int numberOfFolds = 10;
        int numberOfThreads = calculateNumThreads(numberOfFolds);;
        Preprocessor preprocessor = null;
        String[] dataSets = {"BPMFCC"}; // GridSearch para apenas um tipo de dataset
        String[] classifier = {"KNN"}; // GridSearch somente para o KNN


        System.out.println("The machine has " + Runtime.getRuntime().availableProcessors() + " cores processors");
        System.out.println("Using " + numberOfThreads + " threads for parallel execution.");
        for (int kValue : kValueSearch) {
            for (float probabilityMutationSelectInstances : mutationSearch){
                for (float probabilityCrossoverSelectInstances : crossoverSearch){
                    for (int populationSize: populationSearch){
                        for(String runningDataSet : dataSets){
                            for(String runningClassifier : classifier){
                                Map<Integer, Future<Object>> results = new HashMap<>();
                                for(ModelOrganism organism : ModelOrganism.values()){ // GridSearch para todos os organismos
                                    ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
                                    int indexThread = 1;
                                    for(int fold = 0; fold < numberOfFolds; fold++){
                                        try
                                        {
                                            System.out.print("\n##########################################\n");
                                            System.out.println("Reading all the data...");

                                            preprocessor = new Preprocessor(organism, runningDataSet, dataSets, runningClassifier, fold, kValue);

                                            System.out.println("The read have been complete and the data are into memory!");
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

                                    //Testar para a melhor solução gerada
                                    int bestSolutionCounter = 0;
                                    double bestGMean = 0, bestRatioR = 0, auxGMean, auxRatioR;
                                    ArrayList solAndPopulation;
                                    BinarySolution auxSolution;
                                    for(int i = 1; i < (results.size() + 1); i++){
                                        solAndPopulation = (ArrayList) results.get(i).get();
                                        auxSolution =  (BinarySolution) solAndPopulation.get(0);
                                        auxGMean = auxSolution.objectives()[0] *(-1);
                                        auxRatioR = auxSolution.objectives()[1]*(-1);

                                        if(auxGMean > bestGMean || (auxGMean == bestGMean && auxRatioR > bestRatioR)){
                                            bestSolutionCounter = i;
                                            bestGMean = auxGMean;
                                            bestRatioR = auxRatioR;
                                        }
                                    }

                                    List<Instances> trainFold = preprocessor.getDatasetsTRAFolds();
                                    List<Instances> testFold = preprocessor.getDatasetsTESTFolds();

                                    solAndPopulation = (ArrayList) results.get(bestSolutionCounter).get();

                                    double[] resultsClassify = preprocessor.getClassifier().classifySolution( (BinarySolution) solAndPopulation.get(0), trainFold, testFold, true);

                                    String output = "results\\results-gridsearch.csv";
                                    FileHandler.saveResults(output, populationSize, probabilityMutationSelectInstances, probabilityCrossoverSelectInstances, resultsClassify, preprocessor.getOrganism().originalDataset, runningDataSet, bestGMean, preprocessor.getFold(), preprocessor.getKValue(), true);

                                    executor.shutdown();
                                }
                            }
                        }
                    }
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
