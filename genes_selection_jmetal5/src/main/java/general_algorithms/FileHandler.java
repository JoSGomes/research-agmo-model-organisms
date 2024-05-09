/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package general_algorithms;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import com.opencsv.CSVWriter;
import data_processing.ModelOrganism;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author pbexp
 */
public class FileHandler {


    private static CSVWriter csvWriter = null; // Declaração do CSVWriter estático
    // Outros métodos e variáveis da classe...

    public static synchronized void initCSVWriter(String output) throws IOException {
        if (csvWriter == null) {
            Writer writer = new FileWriter(output, true);
            csvWriter = new CSVWriter(writer);
            //Precisa retirar kValue dos headers depois de executar o gridsearch.
            String[] headers = new String[]{"organism", "populationSize", "mutationProbability", "crossoverProbability", "kValue", "runningDataset", "fold", "GMean", "selectionRate", "Best GMean AGMO"};
            csvWriter.writeAll(Collections.singletonList(headers));
            csvWriter.flush();
        }
    }

    public static synchronized void closeCSVWriter() throws IOException {
        if (csvWriter != null) {
            csvWriter.close();
            csvWriter = null;
        }
    }

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

    public static ConcurrentHashMap<String, ConcurrentHashMap<String, ConcurrentHashMap<String, List<Instances>>>> readAllDatasetsFolds(String[] datasets) throws Exception {
        ConcurrentHashMap<String, ConcurrentHashMap<String, ConcurrentHashMap<String, List<Instances>>>> allDatasets = new ConcurrentHashMap<>();

        String typeDataset = "folds";
        int folds = 10;
        for (ModelOrganism o : ModelOrganism.values()) {
            ConcurrentHashMap<String, ConcurrentHashMap<String, List<Instances>>> typeDatasetMap =  new ConcurrentHashMap<>();
                for (String dataset: datasets) {
                    ConcurrentHashMap<String, List<Instances>> datasetListMap = new ConcurrentHashMap<>();
                    String pathDataTra = "";
                    String pathDataTst = "";
                    String pathDataVal = "";
                    List<Instances> listTra = new ArrayList<>();
                    List<Instances> listVal = new ArrayList<>();
                    List<Instances> listTst = new ArrayList<>();
                    for (int n = 0; n < folds; n++){
                        pathDataTra = PathOfDataset.root.path +
                                        o.originalDataset + "\\" +
                                typeDataset + "\\" + dataset + "\\" + o.name().toLowerCase() +
                                        "-" + dataset + "_fold_" + n + "_tra" + FileExtension.arff.extension;

                        pathDataVal = PathOfDataset.root.path +
                                o.originalDataset + "\\" +
                                typeDataset + "\\" + dataset + "\\" +o.name().toLowerCase() +
                                "-" + dataset + "_fold_" + n + "_val" + FileExtension.arff.extension;

                        pathDataTst = PathOfDataset.root.path +
                                o.originalDataset + "\\" +
                                typeDataset + "\\" + dataset + "\\" +o.name().toLowerCase() +
                                "-" + dataset + "_fold_" + n + "_tst" + FileExtension.arff.extension;

                        DataSource sourceTra = new DataSource(pathDataTra);
                        DataSource sourceVal = new DataSource(pathDataVal);
                        DataSource sourceTst = new DataSource(pathDataTst);
                        Instances dataTra = sourceTra.getDataSet();
                        Instances dataVal = sourceVal.getDataSet();
                        Instances dataTst = sourceTst.getDataSet();

                        if(dataTra.classIndex() == -1 ){
                            dataTra.setClassIndex(0);
                        }
                        if(dataVal.classIndex() == -1 ){
                            dataVal.setClassIndex(0);
                        }
                        if(dataTst.classIndex() == -1 ){
                            dataTst.setClassIndex(0);
                        }
                        listTra.add(dataTra);
                        listVal.add(dataVal);
                        listTst.add(dataTst);
                    }



                    datasetListMap.put("tra", listTra);
                    datasetListMap.put("val", listVal);
                    datasetListMap.put("tst", listTst);
                    typeDatasetMap.put(dataset, datasetListMap);

                    allDatasets.put(o.originalDataset, typeDatasetMap);
                }

        }
        return allDatasets;
    }

    /**
     * @param populationSizeA
     * @param mutationA
     * @param crossoverA
     * @param results
     * @param organism
     * @param bestGMean
     */
    public static synchronized void saveResults(int populationSizeA, double mutationA, double crossoverA, double[] results, String organism, String runningDataset, double bestGMean, int fold, int kValue, boolean gridSearch) throws IOException {
        String mutation = Double.toString(mutationA);
        String crossover = Double.toString(crossoverA);
        String populationSize = Integer.toString(populationSizeA);
        String GMean = Double.toString(results[0]);
        String selectionRate = Double.toString(results[1]);
        String[] aux = null;

        if (!gridSearch) {
            aux = new String[]{organism, populationSize, mutation, crossover, runningDataset, Integer.toString(fold), GMean, selectionRate, Double.toString(bestGMean)};
        }
        else {
            aux = new String[]{organism, populationSize, mutation, crossover, String.valueOf(kValue), runningDataset, Integer.toString(fold), GMean, selectionRate, Double.toString(bestGMean)};
        }

        resultsCSV.add(aux);
        writeResults();
    }

    /**
     *
     */
    public static synchronized void writeResults() throws IOException {
        csvWriter.writeAll(resultsCSV);
        csvWriter.flush();
        resultsCSV = new ArrayList<String[]>();
    }

    public static List<String> readOrganismAttributes(Instances data) throws Exception {
        List<String> attributes = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            attributes.add(attribute.name());
        }
        attributes.remove(0);
        return attributes;
    }


    /**
     *
     * @param organism Organismo no qual será realizada a leitura dos ancestrais de cada GO Termo.
     * @return HashMap com chave o nome do GO Termo e valor uma lista de GO Termos.
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static HashMap<String, List<String>> readAncestors(String organism, String runningDataset, boolean desc) throws FileNotFoundException, IOException{
        HashMap<String, List<String>> organismAncestorsGOTerms = new HashMap<>();
        String pathToDAG = "ASC";
        if (desc) {
            pathToDAG = "DESC";
        }

        String file = PathOfDataset.root.path  + organism + "\\DAG\\"+ pathToDAG + "\\" + "DAG-" + runningDataset + "-" + pathToDAG + FileExtension.txt.extension;
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
