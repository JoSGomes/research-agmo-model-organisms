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
    private int folds;
    int cont; //contagem dos bits não selecionados (selection rate)
    private Preprocessor preProcessor;
    private final String runningClassifier; 
    
    /**
     *
     * @param p1
     * @param runningClassifier
     * @throws Exception
     */
    public Classifier(Preprocessor p1, String runningClassifier) throws Exception{
        this.cont = 0;
        this.folds = 10;
        this.preProcessor = p1;
        this.runningClassifier = runningClassifier;
    }
    
    public double[] classifyResult(ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception{      
        switch(this.runningClassifier){
            case "KNN" -> {
                IBk classifier = new IBk(1);
                JaccardDistance jdDist = new JaccardDistance();
                classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
                return calcGMeanSelectionRate(classifier, trainFolds, testFolds);
            }
            case "NB" -> {
                NaiveBayes classifier = new NaiveBayes();
                return calcGMeanSelectionRate(classifier, trainFolds, testFolds);
            }
            case "J48" -> {
                J48 classifier = new J48();
                return this.calcGMeanSelectionRate(classifier, trainFolds, testFolds);
            }
        }       
        return null;
    }
    
    public double[] classifySolution(BinarySolution bestSolution, List<Instances> trainFolds, List<Instances> testFolds) throws Exception{
        switch(this.runningClassifier){
            case "KNN" -> {
                IBk classifier = new IBk(1); //Rever o valor de k...
                JaccardDistance jdDist = new JaccardDistance();
                classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
                List<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
                List<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);
                return calcGMeanSelectionRate(classifier, traData, testData);
            }
            case "NB" -> {
                NaiveBayes classifier = new NaiveBayes();
                List<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
                List<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);
                return calcGMeanSelectionRate(classifier, traData, testData);
            }
            case "J48" -> {
                J48 classifier = new J48();
                List<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
                List<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);
                return calcGMeanSelectionRate(classifier, traData, testData);
            }
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
        double[] GMeans = new double[10];

        Random rData = new Random();
        Instances trainSet, testSet;
        Evaluation eval;

        for(int i=0; i < this.folds; i++) {
            trainSet = tra.get(i);
            testSet = test.get(i);

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

            GMean = (Math.round((Math.sqrt(sensivity * specificity))*10000.00)/10000.00)*100.00;
            GMeans[i] = GMean;
        }

        
        selectionRate =  ( this.preProcessor.getNumAttributes() - cont)
                           / (double) this.preProcessor.getNumAttributes();
        OptionalDouble averageGMean = Arrays.stream(GMeans).average();

        return new double[]{averageGMean.getAsDouble(), selectionRate};
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
    
    /** Método dedicado ao GridSearch, mais otimizado para que não demore
     * muito tempo na execução, tem o mesmo propósito do classificador 
     * original.
     *
     * @param s1 Solução gerada pelo NSGAII.
     * @param tra Instâncias de treino.
     * @param test Instâncias de teste.
     * @return Retorna uma array de tamanho igual a 2, no primeiro está a GMean
     * e na Segunda a Reduction Ratio.
     * @throws Exception
     */
    public double[] classifyKNNGridSearch(BinarySolution s1, List<Instances> tra, List<Instances> test) throws Exception {
        IBk knn = new IBk(1);
	    JaccardDistance jdDist = new JaccardDistance();
	
        knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
        
        List<Instances> traFold = this.getSelectedDatasetFromSolution(s1, tra);
        List<Instances> testFold = this.getSelectedDatasetFromSolution(s1, test);
        
	    return this.calcGMeanSelectionRate(knn, traFold, testFold);
    }

}  
