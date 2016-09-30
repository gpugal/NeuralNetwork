/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.cnn.utility;

import org.jblas.DoubleMatrix;

/**
 *
 * @author pugal
 */
public interface CostFunction {
   
    /**
     * @param a current output 
     * @param y desired output
     * @return @Return the cost associated with an output 'a' and desired output 'y'
     */
    public DoubleMatrix cost(DoubleMatrix a, DoubleMatrix y);
   
    /**
     * @param a current output
     * @param y desired output
     * @param z current layer z-vector
     * @return the error delta from the output layer
     */
    public DoubleMatrix delta(DoubleMatrix a, DoubleMatrix y, DoubleMatrix z);
}
