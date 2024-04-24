/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data_processing;

import java.util.HashMap;
import java.util.List;

import weka.core.Instances;

import general_algorithms.Classifier;
import general_algorithms.FileHandler;

/**
 *
 * @author pbexp
 */
public class Preprocessor {
    private final int numAtributes;
    private final ModelOrganism organism;   
    private Classifier classifier;
    private final String runningDataset;
    private final String runningClassifier;
    private final HashMap<String, HashMap<String, HashMap<String, List<Instances>>>> allDataset;
    private final HashMap<String, List<String>> ascTerms;
    private final HashMap<String, List<String>> descTerms;
    private final HashMap<String, List<String>> megerdADTerms;
    private final List<String> organismAttributes;
    private final int fold;
    private final int kValue;
    /**
     *
     * @param organism
     * @param runningDataSet
     * @param runningClassifier
     * @throws Exception
     */
    public Preprocessor(ModelOrganism organism, String runningDataSet, String[] ableDatasets, String runningClassifier, int fold, int kValue) throws Exception {
        this.allDataset = FileHandler.readAllDatasetsFolds(ableDatasets); // Todos os datasets
        this.runningClassifier = runningClassifier; //NB, KNN, J48
        this.runningDataset = runningDataSet; //BP, MF, CC, BPMF, BPCC, MFCC, BPMFCC
        this.organism = organism; // Caenorhabditis elegans, Drosophila melanogaster, Mus musculus, Saccharomyces cerevisiae
        this.classifier = null;
        this.ascTerms = this.getOrganismAncestors(false);
        this.descTerms = this.getOrganismAncestors(true);
        this.megerdADTerms = this.mergeADTerms();
        this.organismAttributes = this.getOrganismAttributesFromInstance();
        this.fold = fold;
        this.numAtributes = this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tra").get(this.fold).get(0).numAttributes() - 1;
        this.kValue = kValue;
    }
    
    public List<Instances> getTRAFoldAGMO() {
        return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tra");
    }

    public List<Instances> getVALFoldAGMO() {
        return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("val");
    }
    
    public List<Instances> getTESTFoldAGMO() throws Exception{
        return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tst");
    }
    
    public List<Instances> getFoldTRAGridSearch() throws Exception{
        return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tra");
    }
    
    public List<Instances> getFoldTESTGridSearch() throws Exception{
        return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tst");
    }
    
    /** Retorna todos os folds de treino.
     * 
     * @return Lista de Instances
     * @throws Exception
     */
    public List<Instances> getDatasetsTRAFolds() throws Exception{
        return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tra");
    }
    
    /** Retorna todos os folds de teste genuínos.
     *
     * @return Lista de Instances
     * @throws Exception
     */
    public List<Instances> getDatasetsTESTFolds() throws Exception{
         return this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tst");
        
    }

    private HashMap<String, List<String>> getOrganismAncestors(boolean desc) throws Exception {
        return FileHandler.readAncestors(this.organism.originalDataset, this.runningDataset, desc);
    }

    private List<String> getOrganismAttributesFromInstance() throws Exception{
        // Tanto faz se vai ser de treino ou teste, basta ser apenas do organismo e dataset correto (BP, MF...), pois aqui está buscando-se somente os attributos.
        return FileHandler.readOrganismAttributes(this.allDataset.get(this.organism.originalDataset).get(this.runningDataset).get("tra").get(0));
    }

    private HashMap<String, List<String>> mergeADTerms() {
        HashMap<String, List<String>> result = new HashMap<>(this.ascTerms);
        this.descTerms.forEach((key, value) -> result.merge(key, value, (list1, list2) -> {
            list1.addAll(list2);
            return list1;
        }));

        return result;
    }
    /** Retorna o organismo atual que está em curso
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
           classifier = new Classifier(this, this.runningClassifier, this.fold);
       }
       return this.classifier;
    }
    
    /** Retorna o número de atributos do dataset do organismo em questão
     *
     * @return Inteiro que corresponde ao número de atributos.
     *
     */
    public int getNumAttributes(){
        return this.numAtributes;
    }

    public HashMap<String, List<String>> getAscTerms(){
        return this.ascTerms;
    }

    public HashMap<String, List<String>> getDescTerms(){
        return this.descTerms;
    }

    public List<String> getOrganismAttributes(){
        return this.organismAttributes;
    }

    public HashMap<String, List<String>> getMegerdADTerms(){
        return this.megerdADTerms;
    }

    public String getRunningDataset(){
        return this.runningDataset;
    }

    public int getFold(){
        return this.fold;
    }

    public int getKValue(){
        return this.kValue;
    }
}
