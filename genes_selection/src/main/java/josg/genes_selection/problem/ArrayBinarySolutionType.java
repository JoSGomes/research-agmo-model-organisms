package josg.genes_selection.problem;

import jmetal.core.Problem;
import jmetal.core.SolutionType;
import jmetal.core.Variable;
import jmetal.encodings.variable.Binary;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pbexp
 */
public class ArrayBinarySolutionType extends SolutionType{

    private final int binaryStringLength_ ;
    
    public ArrayBinarySolutionType(Problem problem, int binaryStringLenght) {
            super(problem) ;
            binaryStringLength_ = binaryStringLenght;		
    }
    
    
    @Override
    public Variable[] createVariables() throws ClassNotFoundException {
        Variable [] variables = new Variable[1];

        variables[0] = new Binary(binaryStringLength_); 
        return variables ;
    }
    
}
