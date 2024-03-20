/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package general_algorithms;

import com.opencsv.CSVWriter;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import data_processing.ModelOrganism;

import smile.glm.model.Model;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author pbexp
 */
public class FileHandler {
    
    /**
     *
     */
    public static List<String[]> resultsCSV = new ArrayList<String[]>(); 

    /**
     *
     */
    private enum FileExtension {
        txt(".txt"),
        arff(".arff");

        public final String extension;

        FileExtension(String extension){
            this.extension = extension;

        }
    }

    private enum PathOfDataset {
        root("resources\\datasets\\"),
        traAndTest("-dataset-folds\\"),

        rootTRAandTESTAGMO("-dataset-folds\\test-train-agmo\\"),

        foldPath("-fold");

        public final String path;

        PathOfDataset(String path){
            this.path = path;

        }
    }

    /**
     *
     * @param organism
     * @param fold
     * @param runningDataset
     * @return
     * @throws Exception
     */
    public static ArrayList<Instances> readDatasetTRAFoldAGMO(ModelOrganism organism, int fold, String runningDataset) throws Exception
    {
        ArrayList<Instances> foldData = new ArrayList<>();

        String file = PathOfDataset.root + runningDataset + organism.originalDataset
                + PathOfDataset.rootTRAandTESTAGMO + organism.originalDataset + "TRA-" + fold + PathOfDataset.foldPath + FileExtension.arff;

        DataSource source = new DataSource(file);
        Instances data = source.getDataSet();
        if(data.classIndex() == -1 ){
            data.setClassIndex(data.numAttributes() -1);           
        }
        foldData.add(data);
        
        
        return foldData;
    }
    
    /**
     *
     * @param organism
     * @param fold
     * @param runningDataset
     * @return
     * @throws Exception
     */
    public static ArrayList<Instances> readDatasetTESTFoldAGMO(ModelOrganism organism, int fold, String runningDataset) throws Exception
    {
        ArrayList<Instances> foldData = new ArrayList<>();

        String file = PathOfDataset.root.path + runningDataset + organism.originalDataset
                + PathOfDataset.rootTRAandTESTAGMO.path + organism.originalDataset + "TEST-" + fold + PathOfDataset.foldPath.path + FileExtension.arff.extension;

        DataSource source = new DataSource(file);
        Instances data = source.getDataSet();
        if(data.classIndex() == -1 ){
            data.setClassIndex(data.numAttributes() -1);           
        }
        foldData.add(data);
        
        
        return foldData;
    }
    
    /**
     * Retorna o uma ArrayList com apenas um elemento que corresponde ao Fold 
     * especificado de treino  dos dados de teste da melhor solução resultada
     * do AGMO.
     * @param organism Organismo que se deseja extrair os dados.
     * @param fold Fold específico dos dados do organismo.
     * @param runningDataset
     * @return ArrayList com apenas um elemento.
     * @throws Exception
     */
    public static Instances readDatasetTRAFold(ModelOrganism organism, int fold, String runningDataset) throws Exception{
        
        Instances data;

        String file = PathOfDataset.root.path + runningDataset + organism.originalDataset + PathOfDataset.traAndTest.path
                + organism.originalDataset + "TRA-" + fold + PathOfDataset.foldPath.path + FileExtension.arff.extension;

        data = new Instances(new BufferedReader(new FileReader(file)));
        if(data.classIndex() == -1 ){
            data.setClassIndex(data.numAttributes() -1);           
        }
        
        return data;
    }
    
    /**
     * Retorna o uma ArrayList com apenas um elemento que corresponde ao Fold 
     * especificado de teste  dos dados de teste da melhor solução resultada
     * do AGMO.
     * @param organism Organismo que se deseja extrair os dados.
     * @param fold Fold específico dos dados do organismo.
     * @param runningDataset
     * @return ArrayList com apenas um elemento.
     * @throws Exception
     */
    public static ArrayList<Instances> readDatasetTESTFold(ModelOrganism organism, int fold, String runningDataset) throws Exception{
        
        ArrayList<Instances> foldsData = new ArrayList<>();

        String file = PathOfDataset.root.path + runningDataset + organism.originalDataset + PathOfDataset.traAndTest.path
                + organism.originalDataset + "TEST-" + fold + PathOfDataset.foldPath.path + FileExtension.arff.extension;


        DataSource source = new DataSource(file);
        Instances data = source.getDataSet();
        if(data.classIndex() == -1 ){
            data.setClassIndex(data.numAttributes() -1);           
        }
        foldsData.add(data);
        
        
        return foldsData;
    }
    
    /**
     *
     * @param organism
     * @param fold
     * @param runningDataset
     * @return
     * @throws Exception
     */
    public static ArrayList<Instances> readDatasetTRAFolds(ModelOrganism organism, int fold, String runningDataset) throws Exception{
        
        ArrayList<Instances> foldsData = new ArrayList<>();

        String file = PathOfDataset.root.path + runningDataset + organism.originalDataset + PathOfDataset.traAndTest.path
                + organism.originalDataset + "TRA-" + fold + PathOfDataset.foldPath.path + FileExtension.arff.extension;

        DataSource source = new DataSource(file);
        Instances data = source.getDataSet();
        if(data.classIndex() == -1 ){
            data.setClassIndex(data.numAttributes() -1);           
        }
        foldsData.add(data);
        
        
        return foldsData;
    }
    
    /**
     *
     * @param organism
     * @param fold
     * @return
     * @throws Exception
     */
    public static ArrayList<Instances> readDatasetTESTFolds(ModelOrganism organism, int fold, String runningDataset) throws Exception{
        ArrayList<Instances> foldsData = new ArrayList<>();

        String file = PathOfDataset.root.path + runningDataset + organism.originalDataset + PathOfDataset.traAndTest.path
                + organism.originalDataset + "TEST-" + fold + PathOfDataset.foldPath.path + FileExtension.arff.extension;


        DataSource source = new DataSource(file);
        Instances data = source.getDataSet();
        if(data.classIndex() == -1 ){
            data.setClassIndex(data.numAttributes() -1);           
        }
        foldsData.add(data);
        
        return foldsData;
    }
    
    /**
     *
     * @param organism
     * @param folds
     * @param tra
     * @return
     * @throws Exception
     */
    public static ArrayList<Instances> readFoldGridSearch(ModelOrganism organism, int folds, boolean tra) throws Exception {
        ArrayList<Instances> foldData = new ArrayList<>();

        String file = "";
        for(int n =0; n < folds; n++){
            if(tra) {
                file = PathOfDataset.root.path + organism.originalDataset
                          + PathOfDataset.rootTRAandTESTAGMO.path
                          + organism.originalDataset+ "TRA-" + n + PathOfDataset.foldPath.path + FileExtension.arff.extension;
            }
            else {
                file = PathOfDataset.root.path + organism.originalDataset
                          + PathOfDataset.rootTRAandTESTAGMO.path
                          + organism.originalDataset+ "TEST-" + n + PathOfDataset.foldPath.path + FileExtension.arff.extension;
            }
            DataSource source = new DataSource(file);
            Instances fold = source.getDataSet();
            if(fold.classIndex() == -1 ){
                fold.setClassIndex(fold.numAttributes() -1);           
            }
            foldData.add(fold);
        }

        return foldData;
    }
   
    /**
     *
     * @param output
     * @param populationSizeA
     * @param mutationA
     * @param crossoverA
     * @param results
     * @param organism
     * @param fold
     * @param bestGMean
     */
    public static void saveResults(String output, int populationSizeA, double mutationA, double crossoverA, double[] results, String organism, int fold, double bestGMean){
        if(resultsCSV.isEmpty()){
            String[] headers = new String[]{"organism", "populationSize","mutationProbability", "crossoverProbability", "Fold", "GMean" , "Best GMean AGMO", "ratioReduction"};
            resultsCSV.add(headers);
        }

        String mutation = Double.toString(mutationA);
        String crossover = Double.toString(crossoverA);
        String populationSize = Integer.toString(populationSizeA);
        
        double GMean = results[0];
        double ratioReduction = results[1];
        GMean = GMean * (-1);
        ratioReduction = ratioReduction * (-1);
        
        String[] aux = {organism, populationSize, mutation, crossover, Integer.toString(fold), Double.toString(GMean), Double.toString(bestGMean), Double.toString(ratioReduction)};
        resultsCSV.add(aux);
        writeResults(output);
    }
    
    /**
     *
     * @param output Nome da saída
     */
    public static void writeResults(String output){    
        try{ 
            try (Writer writer = new FileWriter(output); CSVWriter csvWriter = new CSVWriter(writer)) {
                csvWriter.writeAll(resultsCSV);
            }
        } catch (IOException ex) {
            Logger.getLogger(FileHandler.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    public static void saveResults(String output, double[] results, String organism, int fold){
        if(resultsCSV.isEmpty()){
            String[] headers = new String[]{"organism", "populationSize","mutationProbability", "crossoverProbability", "Fold", "GMean" , "Best GMean AGMO", "ratioReduction"};
            resultsCSV.add(headers);
        }

        double GMean = results[0];
        String[] aux = {organism, "", "", "", Integer.toString(fold), Double.toString(GMean), "", ""};

        resultsCSV.add(aux);
        writeResults(output);
    }
    
    
    /**
     *
     * @param organism Organismo no qual será realizada a leitura dos ancestrais de cada GO Termo.
     * @return HashMap com chave o nome do GO Termo e valor uma lista de GO Termos.
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static HashMap<String, List<String>> readAncestors(ModelOrganism organism) throws FileNotFoundException, IOException{
        HashMap<String, List<String>> organismAncestorsGOTerms = new HashMap<>();

        String file = PathOfDataset.root.path + "gene_ancestors_" + organism.originalDataset + FileExtension.txt.extension;
        BufferedReader readingFile = new BufferedReader(new FileReader(file));

        String line;
        while((line = readingFile.readLine()) != null){
            String[] terms;
            String mainTerm;
            terms = line.split(" ");
            mainTerm = terms[0];

            List<String> ancestors = new ArrayList<>(Arrays.asList(terms).subList(1, terms.length));
            organismAncestorsGOTerms.put(mainTerm, ancestors);
        } 
        
        return organismAncestorsGOTerms;
    }

}
