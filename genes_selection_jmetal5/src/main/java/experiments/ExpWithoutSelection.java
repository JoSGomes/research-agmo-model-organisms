/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.concurrent.*;

import data_processing.ModelOrganism;
import data_processing.Preprocessor;
import general_algorithms.FileHandler;

import java.util.List;

import weka.core.Instances;

/**
 * @author pbexp
 */
public class ExpWithoutSelection {

    public static void main(String[] args) throws InterruptedException, ExecutionException, Exception, Exception {
        //Controle dos datasets
        int numberOfFolds = 10;

        Preprocessor preprocessor = null;
        String[] dataSets = {"BP", "MF", "CC", "BPMF", "BPCC", "MFCC", "BPMFCC"};
        String[] classifier = {"KNN", "NB", "J48"};

        System.out.println("Reading all the data...");
        ConcurrentHashMap<String, ConcurrentHashMap<String, ConcurrentHashMap<String, List<Instances>>>> allDatasets = FileHandler.readAllDatasetsFolds(dataSets);
        System.out.println("The read have been complete and the data are into memory!");

        for (String runningDataSet : dataSets) {
            for (String runningClassifier : classifier) {
                FileHandler.closeCSVWriter();
                String output = "results\\" + runningClassifier + "\\" + runningDataSet + "\\results-without-selection.csv";
                FileHandler.initCSVWriter(output);
                for (ModelOrganism organism : ModelOrganism.values()) {
                    int kValue = 0;
                    for (int fold = 0; fold < numberOfFolds; fold++) {
                        //Parâmetros do NSGAII
                        switch (organism.originalDataset) {
                            case "Drosophila melanogaster" -> {
                                switch (runningDataSet) {
                                    case "CC" -> {
                                        kValue = 1;
                                    }
                                    case "MF" -> {
                                        kValue = 1;
                                    }
                                    case "MFCC" -> {
                                        kValue = 1;
                                    }
                                    case "BP" -> {
                                        kValue = 1;
                                    }
                                    case "BPCC" -> {
                                        kValue = 1;
                                    }
                                    case "BPMF" -> {
                                        kValue = 5;
                                    }
                                    case "BPMFCC" -> {
                                        kValue = 5;
                                    }
                                }
                            }
                            case "Mus musculus" -> {
                                switch (runningDataSet) {
                                    case "CC" -> {
                                        kValue = 1;
                                    }
                                    case "MF" -> {
                                        kValue = 1;
                                    }
                                    case "MFCC" -> {
                                        kValue = 1;
                                    }
                                    case "BP" -> {
                                        kValue = 1;
                                    }
                                    case "BPCC" -> {
                                        kValue = 1;
                                    }
                                    case "BPMF" -> {
                                        kValue = 5;
                                    }
                                    case "BPMFCC" -> {
                                        kValue = 1;
                                    }
                                }
                            }
                            case "Caenorhabditis elegans" -> {
                                switch (runningDataSet) {
                                    case "CC" -> {
                                        kValue = 1;
                                    }
                                    case "MF" -> {
                                        kValue = 1;
                                    }
                                    case "MFCC" -> {
                                        kValue = 1;
                                    }
                                    case "BP" -> {
                                        kValue = 1;
                                    }
                                    case "BPCC" -> {
                                        kValue = 1;
                                    }
                                    case "BPMF" -> {
                                        kValue = 1;
                                    }
                                    case "BPMFCC" -> {
                                        kValue = 1;
                                    }
                                }
                            }
                            case "Saccharomyces cerevisiae" -> {
                                switch (runningDataSet) {
                                    case "CC" -> {
                                        kValue = 9;
                                    }
                                    case "MF" -> {
                                        kValue = 1;
                                    }
                                    case "MFCC" -> {
                                        kValue = 1;
                                    }
                                    case "BP" -> {
                                        kValue = 1;
                                    }
                                    case "BPCC" -> {
                                        kValue = 1;
                                    }
                                    case "BPMF" -> {
                                        kValue = 1;
                                    }
                                    case "BPMFCC" -> {
                                        kValue = 1;
                                    }
                                }
                            }
                        }
                        preprocessor = new Preprocessor(organism, runningDataSet, allDatasets, runningClassifier, fold, kValue);

                        System.out.println("Starting... " + runningClassifier + " // " + organism.originalDataset + " // " + runningDataSet + " // FOLD - " + fold);

                        List<Instances> trainFolds = preprocessor.getTRAFoldAGMO();
                        List<Instances> testFolds = preprocessor.getTESTFoldAGMO();

                        double[] resultsClassify = preprocessor.getClassifier().classifySolution(null, trainFolds, testFolds);

                        FileHandler.saveResults(0, 0, 0, resultsClassify, preprocessor.getOrganism().originalDataset, preprocessor.getRunningDataset(), 0, preprocessor.getFold(), kValue, false);
                    }
                }
            }
        }
    }
}
