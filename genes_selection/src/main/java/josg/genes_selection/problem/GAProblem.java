/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package josg.genes_selection.problem;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.encodings.solutionType.BinarySolutionType;
import josg.genes_selection.data_processor.Preprocessor;
import josg.genes_selection.general_algorithms.Classifier;


import weka.core.Instances;

/**
 *Objetivos: increse the GMean and reduction rate. 
 * @author pbexp
 */
public class GAProblem extends Problem{  
    
    //private Samples samples_; 
    
    private final Preprocessor preprocessor;
    int folds;
    int seed;
    
    public GAProblem(Preprocessor preprocessor){
        this.preprocessor = preprocessor;
        numberOfVariables_ = preprocessor.getNumAttributes();
        numberOfObjectives_ = 2;
        numberOfConstraints_ = 0;
        problemName_ = "Genes Selection";        
        
        if( solutionType_ == null){
            solutionType_ = new BinarySolutionType(this, numberOfVariables_);
        }
    }
    
    @Override
    public void evaluate(Solution solution) throws jmetal.util.JMException {
        /* Solution: suposta intância que pertence a população criada
        aleatoriamente que deve ser classificada. */
        double[] f = new double[super.getNumberOfObjectives()];    
        double[] GMeanAndReductionRatio;
        try {
            
            GMeanAndReductionRatio = evalF(preprocessor, solution);
            f[0] = GMeanAndReductionRatio[0];
            f[1] = GMeanAndReductionRatio[1];
            
        } catch (Exception ex) {
            Logger.getLogger(GAProblem.class.getName()).log(Level.SEVERE, null, ex);
        }
        solution.setObjective(0, f[0] * (-1));
        solution.setObjective(1, f[1] * (-1));
    }
    public double[] evalF(Preprocessor preprocessor, Solution s1) throws Exception{
        Classifier classifier = preprocessor.getClassifier();

        ArrayList<Instances> tra = preprocessor.getTRAFoldAGMO();
        ArrayList<Instances> test = preprocessor.getTESTFoldAGMO();
        
        double[] results = classifier.classifyKNN(s1, tra, test);
        return results;
    }
   
}

