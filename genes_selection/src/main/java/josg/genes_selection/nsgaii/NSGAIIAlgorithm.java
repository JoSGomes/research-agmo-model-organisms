/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package josg.genes_selection.nsgaii;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.Callable;
import josg.genes_selection.data_processor.Preprocessor;
import josg.genes_selection.problem.GAProblem;
import jmetal.core.*;
import jmetal.core.Problem;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import josg.genes_selection.general_algorithms.FileHandler;
import weka.core.Instances;
/**
 *
 * @author pbexp
 */
public class NSGAIIAlgorithm implements Callable{
    private Problem problem; // The problem to solve
    private Algorithm algorithm; // The algorithm to use

    // operators to SelectInstances problem
    private Operator crossover;
    private Operator mutation;
    private Operator selection;
    private HashMap parameters;
    
    private final Preprocessor preprocessor;
    // parameters AGMO SelectInstances
    private final int populationSizeSelectInstances, maxEvaluationsSelectInstances;
    private final double probabilityCrossoverSelectInstances, probabilityMutationSelectInstances;
    
    int indexThread;
    
    /**
     *
     * @param preprocessor
     * @param populationSizeSelectInstances
     * @param maxEvaluationsSelectInstances
     * @param probabilityCrossoverSelectInstances
     * @param probabilityMutationSelectInstances
     * @param indexThread
     */
    public NSGAIIAlgorithm(Preprocessor preprocessor, int populationSizeSelectInstances, int maxEvaluationsSelectInstances,
                           double probabilityCrossoverSelectInstances, double probabilityMutationSelectInstances, 
                           int indexThread){
        
        this.preprocessor = preprocessor;
        this.populationSizeSelectInstances = populationSizeSelectInstances;
        this.maxEvaluationsSelectInstances = maxEvaluationsSelectInstances;
        this.probabilityCrossoverSelectInstances = probabilityCrossoverSelectInstances;
        this.probabilityMutationSelectInstances = probabilityMutationSelectInstances;
        this.indexThread = indexThread;
    }
    
    @Override
    public Object call() throws Exception {
        try {
            execute();
        } 
            catch (ClassNotFoundException ex) {
            System.err.println("NSGAII class > run method error: " + ex);
            System.exit(-1);
        }
        return 1;
    }
    
        public void execute() throws jmetal.util.JMException, ClassNotFoundException, Exception{          
                problem = new GAProblem(preprocessor);

                algorithm = new NSGAII_Genes_Selection(problem);

                algorithm.setInputParameter("populationSize", populationSizeSelectInstances);
                algorithm.setInputParameter("maxEvaluations", maxEvaluationsSelectInstances);  

                parameters = new HashMap();
                parameters.put("probability", probabilityCrossoverSelectInstances);
                crossover = CrossoverFactory.getCrossoverOperator("HUXCrossover", parameters);

                parameters = new HashMap();
                parameters.put("probability", probabilityMutationSelectInstances);
                mutation = MutationFactory.getMutationOperator("BitFlipMutation", parameters);
                
                parameters = null;
                selection = SelectionFactory.getSelectionOperator("BinaryTournament2", parameters);

                
                
                algorithm.addOperator("mutation", mutation);
                algorithm.addOperator("selection", selection);
                algorithm.addOperator("crossover", crossover);
                
                long initTime = System.currentTimeMillis();
                double estimatedTime;
                double timeSelectInstances;
                //Execute
                SolutionSet subFront = algorithm.execute();
                
                estimatedTime = System.currentTimeMillis() - initTime;
                timeSelectInstances = (estimatedTime/1000) / 60.0;
                System.out.println("Tempo para do Wrapper NSGAII: "+ timeSelectInstances + " minutos");
                
                //Testar para a melhor solução gerada
                
                int bestSolutionCounter = 0;
                double bestGMean = 0, aux;
                for(int i = 0; i < subFront.size(); i++){
                    aux = subFront.get(i).getObjective(0)*(-1);
                    if(aux > bestGMean){
                        bestSolutionCounter = i;
                        bestGMean = aux;
                    }         
                }
                
                Solution bestSolution = subFront.get(bestSolutionCounter); 
                
                ArrayList<Instances> trainFold = preprocessor.getDatasetsTRAFolds();                  
                ArrayList<Instances> testFold = preprocessor.getDatasetsTESTFolds();
                
                double[] results = preprocessor.getClassifier().classifyKNN(bestSolution, trainFold, testFold);
                
                FileHandler.saveResults(algorithm, results, preprocessor.getOrganism().originalDataset, preprocessor.getFold(), bestGMean);  
                subFront.printFeasibleFUN("results\\FUN-" + preprocessor.getOrganism().originalDataset + "\\fold-" + preprocessor.getFold());
                subFront.printFeasibleVAR("results\\VAR-" + preprocessor.getOrganism().originalDataset + "\\fold-" + preprocessor.getFold());
        }
    }

