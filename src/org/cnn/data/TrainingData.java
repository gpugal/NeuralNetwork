/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.cnn.data;

public class TrainingData {
    
    private final int SIZE;

    private double[][] INPUTS;
    private double[][] OUTPUTS;

    public TrainingData(double[][] inputs, double[][] outputs) {
        this.INPUTS = inputs;
        this.OUTPUTS = outputs;
        SIZE = inputs.length;
    }

    public double[] getInputs(int i) {
        return INPUTS[i];
    }

    public double[] getOutputs(int i) {
        return OUTPUTS[i];
    }
    
    public int size() {
        return SIZE;
    }
}
