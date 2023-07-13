/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package josg.genes_selection.data_processor;

import java.util.ArrayList;
import josg.genes_selection.general_algorithms.Classifier;
import josg.genes_selection.general_algorithms.FileHandler;
import weka.core.Instances;

/**
 *
 * @author pbexp
 */
public class Preprocessor{
    
    private final int numAtributes;
    private final int fold;
    private final ModelOrganism organism;   
    private Classifier classifier;
    
    
    /**
     *
     * @param organism
     * @param fold
     * @throws Exception
     */
    public Preprocessor(ModelOrganism organism, int fold) throws Exception{
        this.fold = fold;
        this.numAtributes = FileHandler.readDatasetTESTFoldAGMO(organism, fold).get(0).numAttributes() - 1;
        this.organism = organism;
        classifier = null;
        
    }
    
    public ArrayList<Instances> getTRAFoldAGMO() throws Exception{
        return FileHandler.readDatasetTRAFoldAGMO(organism, fold);
    }
    
    public ArrayList<Instances> getTESTFoldAGMO() throws Exception{    
        return FileHandler.readDatasetTESTFoldAGMO(organism, fold);  
    }
    
    public ArrayList<Instances> getFoldTRAGridSearch() throws Exception{
        ArrayList<Instances> traSet = FileHandler.readFoldGridSearch(organism, fold, true);
        return traSet;  
    }
    
    public ArrayList<Instances> getFoldTESTGridSearch() throws Exception{
        ArrayList<Instances> testSet = FileHandler.readFoldGridSearch(organism, fold, false);
        return testSet;  
    }
    
    /** Retorna o fold de treino genuínos.
     * 
     * @return ArrayList com apenas um elemento.
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTRAFold() throws Exception{
        return FileHandler.readDatasetTRAFold(organism, fold);
        
    }
    
    /** Retorna o fold de teste genuínos.
     *
     * @return ArrayList com apenas um elemento.
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTESTFold() throws Exception{
         return FileHandler.readDatasetTESTFold(organism, fold);
        
    }
    
    /** Retorna todos os folds de treino.
     * 
     * @return Retorna uma lista objetos do tipo Instances (Weka Class)
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTRAFolds() throws Exception{
        return FileHandler.readDatasetTRAFolds(organism, fold);
        
    }
    
    /** Retorna todos os folds de teste genuínos.
     *
     * @return Retorna uma lista objetos do tipo Instances (Weka Class)
     * @throws Exception
     */
    public ArrayList<Instances> getDatasetsTESTFolds() throws Exception{
         return FileHandler.readDatasetTESTFolds(organism, fold);
        
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
     */
    public Classifier getClassifier() throws Exception {
       if(classifier == null){
           classifier = new Classifier(this);
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
