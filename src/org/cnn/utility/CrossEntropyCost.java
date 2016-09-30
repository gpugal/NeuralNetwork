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
 * @date 30 Sep, 2016 - 3:23:38 PM
 */
public class CrossEntropyCost implements CostFunction {

    @Override
    public DoubleMatrix cost(DoubleMatrix a, DoubleMatrix y) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public DoubleMatrix delta(DoubleMatrix a, DoubleMatrix y, DoubleMatrix z) {
        return a.sub(y);
    }
}
