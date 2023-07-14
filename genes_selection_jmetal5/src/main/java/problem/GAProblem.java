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
    private final Preprocessor preprocessor;
    int folds;
    int seed;
    int numberOfVariables;
    int numberOfObjectives;
    int numberOfConstraints;
    String name;
    public GAProblem(Preprocessor preprocessor)
    {
        this.preprocessor = preprocessor;
        this.numberOfVariables = preprocessor.getNumAttributes();
        this.numberOfObjectives = 2;
        this.numberOfConstraints = 0;
        this.name = "Genes Selection";
        
        bitsPerVariable = new ArrayList<>(this.numberOfVariables) ;

        for (int var = 0; var < this.numberOfVariables; var++) {
          bitsPerVariable.add(1);
        }
    }
    @Override
    public int numberOfVariables() {
        return this.numberOfVariables;
    }

    @Override
    public int numberOfObjectives() {
        return this.numberOfObjectives;
    }

    @Override
    public int numberOfConstraints() {
        return this.numberOfConstraints;
    }

    @Override
    public String name() {
        return this.name;
    }

    @Override
    public BinarySolution evaluate(BinarySolution solution) {
        /* Solution: suposta intância que pertence a população criada
        aleatoriamente que deve ser classificada. */
        double[] f = new double[this.numberOfObjectives];
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

    @Override
    public List<Integer> listOfBitsPerVariable() {
        return null;
    }
}

