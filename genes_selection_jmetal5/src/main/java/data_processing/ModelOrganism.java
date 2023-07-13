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
    Fly("fly"),
    Mouse("mouse"),
    Worm("worm"),
    Yeast("yeast");
	
    public String originalDataset;
    
    private ModelOrganism(String originalDataset){
        this.originalDataset = originalDataset;
        
    }
       
}
    
