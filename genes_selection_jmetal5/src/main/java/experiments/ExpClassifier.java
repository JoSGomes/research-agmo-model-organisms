/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import weka.core.Instances;

import data_processing.ModelOrganism;
import data_processing.Preprocessor;
import general_algorithms.Classifier;
import general_algorithms.FileHandler;

/**
 *
 * @author pbexp
 */
public class ExpClassifier {
    
    
    public static void main(String[] args) throws InterruptedException, ExecutionException, Exception{
        int folds = 10;
        for(ModelOrganism organism : ModelOrganism.values())
        {
            for(int i = 0; i < folds; i++){
                Preprocessor p1 = new Preprocessor(organism, i);
                Classifier c1 = p1.getClassifier();
                
                ArrayList<Instances> tra = p1.getDatasetsTRAFolds();
                ArrayList<Instances> test = p1.getDatasetsTESTFolds();
                
                double[] results = c1.classifyJ48(tra, test);
                
                FileHandler.saveResults(results, organism.originalDataset, i);
            }
            
            
            
        }
        
        
        
    }
}
