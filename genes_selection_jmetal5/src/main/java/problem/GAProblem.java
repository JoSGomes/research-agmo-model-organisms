/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package problem;

import java.util.List;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.uma.jmetal.problem.binaryproblem.impl.AbstractBinaryProblem;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import weka.core.Instances;

import data_processing.Preprocessor;
import general_algorithms.Classifier;





/**
 *Objetivos: increse the GMean and reduction rate. 
 * @author pbexp
 */
public class GAProblem extends AbstractBinaryProblem{  
    private List<Integer> bitsPerVariable ;
    //private Samples samples_; 
    
    private final Preprocessor preprocessor;
    int folds;
    int seed;
    
    public GAProblem(Preprocessor preprocessor)
    {
        this.preprocessor = preprocessor;
        setNumberOfVariables(preprocessor.getNumAttributes());
        setNumberOfObjectives(2);
        setNumberOfConstraints(0);
        setName("Genes Selection");        
        
        bitsPerVariable = new ArrayList<>(getNumberOfVariables()) ;

        for (int var = 0; var < getNumberOfVariables(); var++) {
          bitsPerVariable.add(1);
        }
    }
    
    @Override
    public List<Integer> getListOfBitsPerVariable() {
        return this.bitsPerVariable;
    }
    
    @Override
    public BinarySolution evaluate(BinarySolution solution) {
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
        solution.objectives()[0] = f[0] * (-1);
        solution.objectives()[1] = f[1] * (-1);
        
        return solution;
    }
    public double[] evalF(Preprocessor preprocessor, BinarySolution s1) throws Exception{
        Classifier classifier = preprocessor.getClassifier();

        ArrayList<Instances> tra = preprocessor.getTRAFoldAGMO();
        ArrayList<Instances> test = preprocessor.getTESTFoldAGMO();
        
        double[] results = classifier.classifySolution(s1, tra, test);
        return results;
    }
}

