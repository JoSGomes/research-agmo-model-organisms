/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nsgaii;


import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.List;

import problem.GAProblem;
import data_processing.Preprocessor;

import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAII;
import org.uma.jmetal.problem.Problem;

import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.SelectionOperator;
import org.uma.jmetal.operator.selection.impl.BinaryTournamentSelection;
import org.uma.jmetal.operator.mutation.impl.BitFlipMutation;
import org.uma.jmetal.operator.crossover.impl.HUXCrossover;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
/**
 *
 * @author pbexp
 */
public class NSGAIIAlgorithm implements Callable{
    Problem<BinarySolution> problem;
    NSGAII<BinarySolution> algorithm;
    
// operators to SelectInstances problem
    CrossoverOperator<BinarySolution> crossover;
    MutationOperator<BinarySolution> mutation;
    SelectionOperator<List<BinarySolution>, BinarySolution> selection;
    
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
    public ArrayList call() throws Exception {
        ArrayList solAndPopulation = null;
        try {
            solAndPopulation = execute();
        } 
        catch (Exception ex) {
            System.err.println("NSGAII class > run method error: " + ex);
            System.exit(-1);
        }
        return solAndPopulation;
    }
    
    public ArrayList execute() throws Exception, Exception
    {          
            problem = new GAProblem(preprocessor);

            crossover = new HUXCrossover(probabilityCrossoverSelectInstances);
            mutation = new BitFlipMutation(probabilityMutationSelectInstances);     
            selection = new BinaryTournamentSelection();

            algorithm = new NSGAIIBuilder<>(problem, crossover, mutation, populationSizeSelectInstances)
                    .setSelectionOperator(selection)
                    .setMaxEvaluations(maxEvaluationsSelectInstances)
                    .setVariant(NSGAIIBuilder.NSGAIIVariant.NSGAII)
                    .build() ;

            long initTime = System.currentTimeMillis();
            double estimatedTime;
            double timeSelectInstances;

            //Execute
            algorithm.run();
            List<BinarySolution> population = algorithm.getResult();

            estimatedTime = System.currentTimeMillis() - initTime;
            timeSelectInstances = (estimatedTime/1000) / 60.0;
            System.out.println("Tempo para do Wrapper NSGAII: "+ timeSelectInstances + " minutos");

            //Testar para a melhor solução gerada    
            int bestSolutionCounter = 0;
            double bestGMean = 0, bestRatioR = 0, auxGMean, auxRatioR;
            for(int i = 0; i < population.size(); i++){
                auxGMean = population.get(i).objectives()[0]*(-1);
                auxRatioR = population.get(i).objectives()[1]*(-1);
                if(auxGMean > bestGMean || (auxGMean == bestGMean && auxRatioR > bestRatioR)){
                    bestSolutionCounter = i;
                    bestGMean = auxGMean;
                    bestRatioR = auxRatioR;
                }         
            }

            BinarySolution bestSolution = population.get(bestSolutionCounter); 
            ArrayList solAndPopulation = new ArrayList();
            solAndPopulation.add(bestSolution);
            solAndPopulation.add(population);
            
            return solAndPopulation;  
    }
}

