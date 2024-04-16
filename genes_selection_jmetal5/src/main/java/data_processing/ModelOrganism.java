/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data_processing;

/**
 *
 * @author pbexp
 */
public enum ModelOrganism {
    Fly("Drosophila melanogaster"),
    Mouse("Mus musculus"),
    Worm("Caenorhabditis elegans"),
    Yeast("Saccharomyces cerevisiae");
	
    public final String originalDataset;
    
    private ModelOrganism(String originalDataset){
        this.originalDataset = originalDataset;
        
    }
       
}
    
