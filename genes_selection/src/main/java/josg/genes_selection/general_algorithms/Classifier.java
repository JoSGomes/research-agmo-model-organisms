/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package josg.genes_selection.general_algorithms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import josg.genes_selection.data_processor.Preprocessor;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import jmetal.core.Solution;
import jmetal.encodings.variable.Binary;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


/**
 *
 * @author pbexp
 */
public class Classifier {
    private int seed = 1;
    private int folds = 10;
    int cont; //contagem dos bits selecionados (ratio reduction)
    private Preprocessor preProcessor;
    
    /**
     *
     * @param p1
     * @throws Exception
     */
    public Classifier(Preprocessor p1) throws Exception{
        this.cont = 0;
        preProcessor = p1;
    }
    /*Para retirar os resultados sem o AGMO*/
    public double[] classifyKNN(ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception{
        IBk knn = new IBk(1);
        
        JaccardDistance jdDist = new JaccardDistance();
        knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
        
	return this.calcGMeanRatioReduction(knn, trainFolds, testFolds);
    }
    
    public double[] classifyNB(ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception {
	NaiveBayes naive = new NaiveBayes();
             
        return calcGMeanRatioReduction(naive, trainFolds, testFolds);
    }
    
    public double[] classifyJ48(ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception{
        J48 j48 = new J48();
        
	return this.calcGMeanRatioReduction(j48, trainFolds, testFolds);
    }
    
    public double[] classifyJ48(Solution bestSolution, ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception{
        J48 j48 = new J48();
        
        ArrayList<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
        ArrayList<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);

	return this.calcGMeanRatioReduction(j48, traData, testData);
    }
    
    public double[] classifyKNN(Solution bestSolution, ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception {
        IBk knn = new IBk(1);	
        
	JaccardDistance jdDist = new JaccardDistance();
        
	
        knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
        
        ArrayList<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
        ArrayList<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);

	return this.calcGMeanRatioReduction(knn, traData, testData);
    }
    
    public double[] classifyNB(Solution bestSolution, ArrayList<Instances> trainFolds, ArrayList<Instances> testFolds) throws Exception {
	NaiveBayes naive = new NaiveBayes();
        
        ArrayList<Instances> traData = this.getSelectedDatasetFromSolution(bestSolution, trainFolds);
        ArrayList<Instances> testData = this.getSelectedDatasetFromSolution(bestSolution, testFolds);
        
        return calcGMeanRatioReduction(naive, traData, testData);
    }
       
    
    /**Calcula a GMean da Specificity e Sensibility e a Reduction Ratio.
     *
     * @param selectedData Dataset que o modelo será treinado a fim de 
     * verificar a solução.
     * @param classifier Classificador utilizando como configuração de cálculo
     * de Distância a de Jaccard.
     * @return Retorna a GMean e a Reduction Ratio.
     * @throws Exception
     */
    private double[] calcGMeanRatioReduction(AbstractClassifier classifier, ArrayList<Instances> tra, ArrayList<Instances> test) throws Exception {
        double truePos, trueNeg, falsePos, falseNeg;
        double sensivity, specificity;
        double reductionRatio;
        double GMean;
        Random rData = new Random();
        Instances trainSet, testSet;
        Evaluation eval;
                  
        trainSet = tra.get(0);
        testSet = test.get(0);
        
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
        
        reductionRatio =  ( this.preProcessor.getNumAttributes() - cont) 
                           / (double) this.preProcessor.getNumAttributes();
        
        
        double[] results = {GMean, reductionRatio};
        return results;
    }

    /**
     *
     * @param s1
     * @param dataSetFolds
     * @return
     * @throws IOException
     * @throws Exception
     */
    private ArrayList<Instances> getSelectedDatasetFromSolution(Solution s1, ArrayList<Instances> dataSetFolds) throws IOException, Exception{
        cont = 0;
        int bits = s1.getNumberOfBits();
        Binary sol = (Binary) s1.getDecisionVariables()[0];
        ArrayList<Integer> dellAttributes = new ArrayList<>();
         
        for(int i = 0; i < bits; i++)
        {
            if(sol.getIth(i) == true)
            {
                cont++;
            }
            else
            {
                dellAttributes.add(i);  
            }  
            
        } 
        Object[] indicesObject = dellAttributes.toArray();
        int length = indicesObject.length;
        int[] indicesArray = new int[length];
        for(int n = 0; n < length; n++)
        {
            indicesArray[n] = (int) indicesObject[n];
        }
        
        return deleteAttributes(dataSetFolds, indicesArray);
    }  
    
    private ArrayList<Instances> deleteAttributes(ArrayList<Instances> selectedDatasetFolds, int[] indices) throws Exception 
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
        
    /**
     *
     * @param newSeed
     */
    public void setSeed(int newSeed){
        seed = newSeed;
    }
    
    /**
     *
     * @return
     */
    public int getSeed(){
        return this.seed;
    }
    
    /**
     *
     * @param newFolds
     */
    public void setFolds(int newFolds){
        folds = newFolds;
    }
   
    /**
     *
     * @return
     */
    public int getFolds(){
        return this.folds;
    }
    
    /**
     *
     * @param newPreprocessor
     */
    public void setPreProcessor(Preprocessor newPreprocessor){
        preProcessor = newPreprocessor;
    }

    /**
     *
     * @return
     */
    public Preprocessor getPreProcessor(){
        return this.preProcessor;
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
    public double[] classifyKNNGridSearch(Solution s1, ArrayList<Instances> tra, ArrayList<Instances> test) throws Exception {
        IBk knn = new IBk(1);
	JaccardDistance jdDist = new JaccardDistance();
	
        knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(jdDist);
        
        ArrayList<Instances> traFold = this.getSelectedDatasetFromSolution(s1, tra);
        ArrayList<Instances> testFold = this.getSelectedDatasetFromSolution(s1, test);
        
	return this.crossValidationGridSearch(knn, traFold, testFold);
        
    }

    /**Calcula a GMean da Specificity e Sensibility e a Reduction Ratio.
     * 
     * @param selectedData Dataset que o modelo será treinado a fim de 
     * verificar a solução.
     * @param knn Classificador utilizando como configuração de cálculo
     * de Distância a de Jaccard.
     * @return Retorna a GMean e a Reduction Ratio.
     * @throws Exception
     */
    private double[] crossValidationGridSearch(AbstractClassifier classifier, ArrayList<Instances> tra, ArrayList<Instances> test) throws Exception {
        double truePos, trueNeg, falsePos, falseNeg, totalSpe = 0, totalSen = 0;
        double sensivity; 
        double specificity;
        double reductionRatio;
        double GMean;
        int foldNumber;
        Random rData = new Random();
        
        Instances traSet, testSet;
        Evaluation eval;
        
        //foldNumber = rData.nextInt(9);
        for(int n = 0; n < folds; n++)
        {
            traSet = tra.get(n);       
            testSet = test.get(n);

            traSet.randomize(rData);
            testSet.randomize(rData);

            eval = new Evaluation(traSet);   
            classifier.buildClassifier(traSet);
            eval.evaluateModel(classifier, testSet);

            truePos = eval.numTruePositives(1);
            trueNeg = eval.numTrueNegatives(1);
            falsePos = eval.numFalsePositives(1);
            falseNeg = eval.numFalseNegatives(1);

            sensivity = (truePos / (truePos + falseNeg))/Math.pow(10,1); 
            specificity = (trueNeg / (trueNeg + falsePos))/Math.pow(10,1); 
            
            totalSpe += specificity;
            totalSen += sensivity;
        }
        
        
        reductionRatio =  ( this.preProcessor.getNumAttributes() - cont) 
                           / (double) this.preProcessor.getNumAttributes();
        
        GMean = (Math.round((Math.sqrt(totalSpe * totalSen))*10000.00)/10000.00)*100.00;

        double[] results = {GMean, reductionRatio};
        return results;
    }

    

}  
