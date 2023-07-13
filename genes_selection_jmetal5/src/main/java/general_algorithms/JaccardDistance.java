/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package general_algorithms;

import java.io.Serializable;
import java.util.Enumeration;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionHandler;
import weka.core.neighboursearch.PerformanceStats;


/**
 * Classe da Distância de Jaccard para ser usada no classificador KNN, visto que
 * os dados são dispostos de forma binária.
 * 
 * Todos os Créditos deste código a Luan Rios Campos.
 * Gitlab: https://gitlab.com/rcluan.
 * @author pbexp
 */
public class JaccardDistance implements DistanceFunction, OptionHandler, Serializable, RevisionHandler{

    
        private static final long serialVersionUID = 1L;
            
	public static final int R_MIN = 0;
	public static final int R_MAX = 1;
	public static final int R_WIDTH = 2;

        protected Instances m_Data = null;
	protected double[][] m_Ranges;
	protected Range m_AttributeIndices = new Range("first-last");
        
        protected boolean[] m_ActiveIndices;
        
        public JaccardDistance(){

        }
        public JaccardDistance(Instances data){
            this.setInstances(data);
        }
        
        @Override
	public double distance(Instance first, Instance second) {
		
		return distance(first, second, null);
	}

	@Override
	public double distance(Instance arg0, Instance arg1, PerformanceStats arg2) {
		
		return distance(arg0, arg1, Double.POSITIVE_INFINITY, arg2);
	}

	@Override
	public double distance(Instance arg0, Instance arg1, double arg2) {
		
		return distance(arg0, arg1, arg2, null);
	}

	@Override
	public double distance(Instance first, Instance second, double arg2, PerformanceStats arg3) {
		
		double coefficient = 0, m11 = 0, m10 = 0, m01 = 0;
		
		for(int i = 0; i < first.numAttributes(); i++){
			
			if(first.value(i) == 1 && second.value(i) == 1)
				m11 += 1;
			
			if(first.value(i) == 1 && second.value(i) == 0)
				m10 += 1;
			
			if(first.value(i) == 0 && second.value(i) == 1)
				m01 += 1;
		}
		
		
		coefficient = m11/(m11+m10+m01);
		
		return 1 - coefficient;
	}
	

	@Override
	public void setInstances(Instances arg0) {
		
		m_Data = arg0;
	}
	
	
	// Unused methods

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Enumeration<Option> listOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setOptions(String[] arg0) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void clean() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getAttributeIndices() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Instances getInstances() {
		
		return m_Data;
	}

	@Override
	public boolean getInvertSelection() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void postProcessDistances(double[] arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setAttributeIndices(String arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setInvertSelection(boolean arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update(Instance arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}

    
}
