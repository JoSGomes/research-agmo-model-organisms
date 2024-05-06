/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package general_algorithms;


import java.io.IOException;
import java.util.*;

import data_processing.Preprocessor;

import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.util.binarySet.BinarySet;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;

/**
 *
 * @author pbexp
 */
public class Classifier {
    private int fold;
    int cont; //contagem dos bits não selecionados (selection rate)
    private Preprocessor preProcessor;
    private final String runningClassifier; 
    
    /**
     *
     * @param p1
     * @param runningClassifier
     * @throws Exception
     */
    public Classifier(Preprocessor p1, String runningClassifier, int fold) throws Exception{
        this.cont = 0;
        this.fold = fold;
        this.preProcessor = p1;
        this.runningClassifier = runningClassifier;
    }

    public double[] classifySolution(BinarySolution bestSolution, List<Instances> trainFolds, List<Instances> testFolds) throws Exception{
        AbstractClassifier classifier = null;
        switch(this.runningClassifier){
            case "KNN" -> {
                classifier = new IBk(1); //Rever o valor de k...
                JaccardDistance jdDist = new JaccardDistance();
                ((IBk) classifier).getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
            }
            case "NB" -> classifier = new NaiveBayes();
            case "J48" -> classifier = new J48();

        }
        if (classifier != null){
            List<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
            List<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);
            return calcGMeanSelectionRate(classifier, traData, testData);
        }
        return null;
    }

    /**Calcula a GMean da Specificity e Sensibility e a Reduction Ratio.
     *
     * @param tra Dataset que o modelo será treinado a fim de
     * verificar a solução.
     * @param classifier Classificador utilizando como configuração de cálculo
     * de Distância a de Jaccard.
     * @return Retorna a GMean e a Reduction Ratio.
     */
    private double[] calcGMeanSelectionRate(AbstractClassifier classifier, List<Instances> tra, List<Instances> test) throws Exception {
        double truePos, trueNeg, falsePos, falseNeg;
        double sensivity, specificity;
        double selectionRate;
        double GMean;

        Random rData = new Random();
        Instances trainSet, testSet;
        Evaluation eval;

        trainSet = tra.get(this.fold);
        testSet = test.get(this.fold);

        trainSet.randomize(rData);
        testSet.randomize(rData);

        eval = new Evaluation(trainSet);
        classifier.buildClassifier(trainSet);
        eval.evaluateModel(classifier, testSet);

        truePos = eval.numTruePositives(1);
        trueNeg = eval.numTrueNegatives(1);
        falsePos = eval.numFalsePositives(1);
        falseNeg = eval.numFalseNegatives(1);

        sensivity = truePos / (truePos + falseNeg);
        specificity = trueNeg / (trueNeg + falsePos);

        GMean = Math.sqrt(sensivity * specificity);
        
        selectionRate =  ( this.preProcessor.getNumAttributes() - cont)
                           / (double) this.preProcessor.getNumAttributes();

        return new double[]{GMean, selectionRate};
    }

    /**
     *
     * @param s1
     * @param dataSetFolds
     * @return
     * @throws IOException
     * @throws Exception
     */
    private List<Instances> getSelectedDatasetFromSolution(BinarySolution s1, List<Instances> dataSetFolds) throws IOException, Exception{
        cont = 0;
        int bits = s1.totalNumberOfBits();
        List<BinarySet> sol = s1.variables();
        List<Integer> dellAttributes = new ArrayList<>();
         
        for(int i = 0; i < bits; i++)
        {
            if(!sol.get(i).get(0))
            {
                cont++;
                dellAttributes.add(i);
            }
        } 
        Object[] indicesObject = dellAttributes.toArray();
        int length = indicesObject.length;
        int[] indicesArray = new int[length];
        for(int n = 0; n < length; n++)
        {
            indicesArray[n] = (int) indicesObject[n] + 1;
        }
        
        return deleteAttributes(dataSetFolds, indicesArray);
    }  
    
    private List<Instances> deleteAttributes(List<Instances> selectedDatasetFolds, int[] indices) throws Exception
    {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(indices);
        removeFilter.setInputFormat(selectedDatasetFolds.get(0));
        int size = selectedDatasetFolds.size();
        ArrayList<Instances> newData = new ArrayList<>();
        
        for(int n = 0; n < size; n++)
        {
           newData.add(Filter.useFilter(selectedDatasetFolds.get(n), removeFilter));
        }
        
        return newData;
    }

}
