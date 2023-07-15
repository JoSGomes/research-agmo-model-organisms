/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data_processing;

import java.util.ArrayList;

import weka.core.Instances;

import general_algorithms.Classifier;
import general_algorithms.FileHandler;

/**
 *
 * @author pbexp
 */
public class Preprocessor {
    private final int numAtributes;
    private final int fold;
    private final ModelOrganism organism;   
    private Classifier classifier;
    private final String runningDataset;
    private final String runningClassifier;

    /**
     *
     * @param organism
     * @param fold
     * @param runningDataSet
     * @param runningClassifier
     * @throws Exception
     */
    public Preprocessor(ModelOrganism organism, int fold, String runningDataSet, String runningClassifier) throws Exception{
        this.fold = fold;
        this.numAtributes = FileHandler.readDatasetTESTFoldAGMO(organism, fold, runningDataSet).get(0).numAttributes() - 1;
        this.organism = organism;
        this.runningDataset = runningDataSet;
        this.classifier = null;
        this.runningClassifier = runningClassifier;
        
    }
    
    public ArrayList<Instances> getTRAFoldAGMO() throws Exception{
        return FileHandler.readDatasetTRAFoldAGMO(organism, fold, this.runningDataset);
    }
    
    public ArrayList<Instances> getTESTFoldAGMO() throws Exception{    
        return FileHandler.readDatasetTESTFoldAGMO(organism, fold, this.runningDataset);  
    }
    
    public ArrayList<Instances> getFoldTRAGridSearch() throws Exception{
        return FileHandler.readFoldGridSearch(organism, fold, true);
    }
    
    public ArrayList<Instances> getFoldTESTGridSearch() throws Exception{
        return FileHandler.readFoldGridSearch(organism, fold, false);
    }
    
    /** Retorna o fold de treino genuínos.
     * 
     * @return ArrayList com apenas um elemento.
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTRAFold() throws Exception{
        return FileHandler.readDatasetTRAFold(organism, fold, this.runningDataset);
        
    }
    
    /** Retorna o fold de teste genuínos.
     *
     * @return ArrayList com apenas um elemento.
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTESTFold() throws Exception{
         return FileHandler.readDatasetTESTFold(organism, fold, this.runningDataset);
        
    }
    
    /** Retorna todos os folds de treino.
     * 
     * @return Retorna uma lista objetos do tipo Instances (Weka Class)
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTRAFolds() throws Exception{
        return FileHandler.readDatasetTRAFolds(organism, fold, this.runningDataset);
        
    }
    
    /** Retorna todos os folds de teste genuínos.
     *
     * @return Retorna uma lista objetos do tipo Instances (Weka Class)
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTESTFolds() throws Exception{
         return FileHandler.readDatasetTESTFolds(organism, fold, this.runningDataset);
        
    }
    
    /**Retorna o organismo atual que está em curso
     *
     * @return Retorna um enum ModelOrganism com o organismo atual em curso.
     */
    public ModelOrganism getOrganism(){
        return this.organism;
    }

    /** Retorna o objeto da classe Classifier.
     *
     * @return Classifier com todos os métodos de classificação.
     * @throws java.lang.Exception
     */
    public Classifier getClassifier() throws Exception {
       if(classifier == null){
           classifier = new Classifier(this, this.runningClassifier);
       }
       return this.classifier;
    }
    
    /** Retorna o número de atributos do dataset do organismo em questão
     *
     * @return Inteiro que corresponde ao número de atributos.
     */
    public int getNumAttributes(){
        return this.numAtributes;
    }
    
    public int getFold(){
        return this.fold;
    }
}
