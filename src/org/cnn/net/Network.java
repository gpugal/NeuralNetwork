/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.cnn.net;

import org.cnn.utility.Sigmoid;
import org.cnn.data.TrainingData;
import org.cnn.data.DigitDataGenerator;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.cnn.utility.CostFunction;
import org.cnn.utility.QuadraticCost;
import org.jblas.DoubleMatrix;

/**
 *
 * @author pugal
 * @date 9 Sep, 2016 - 9:59:13 PM
 */
public class Network {

    private final int NUM_LAYER;
    private final List<DoubleMatrix> WEIGHTS;
    private final List<DoubleMatrix> BIASES;
    private LinkedList<DoubleMatrix> wDelta;
    private LinkedList<DoubleMatrix> bDelta;
    private CostFunction cost;

    public Network(int size[]) {
        this(size, new QuadraticCost());
    }
    
    public Network(int size[], CostFunction cost) {
        NUM_LAYER = size.length;
        WEIGHTS = new ArrayList<>(NUM_LAYER - 1);
        this.cost = cost;

        for (int i = 0; i < NUM_LAYER - 1; ++i) {
            int n = size[i + 1];
            int m = size[i];
            WEIGHTS.add(DoubleMatrix.randn(n, m));
        }
        BIASES = new ArrayList<>(NUM_LAYER - 1);

        for (int i = 1; i < NUM_LAYER; ++i) {
            int m = size[i];
            BIASES.add(DoubleMatrix.randn(m, 1));
        }
    }
    
    
    
    private DoubleMatrix feedForward(DoubleMatrix x) {
        DoubleMatrix a = x;
        for (int i = 0; i < NUM_LAYER - 1; ++i) {
            DoubleMatrix z = WEIGHTS.get(i).mmul(a).addi(BIASES.get(i));
            a = Sigmoid.apply(z);
        }
        return a;
    }
    
    private int evaluate(DigitDataGenerator testDG) {
         testDG.shuffle();
         int size = testDG.getDataSetSize();
         int sum = 0;
         TrainingData testData = testDG.getTrainingData(size);         
         for (int i = 0; i < testData.size(); ++i) {
             DoubleMatrix x = new DoubleMatrix(testData.getInputs(i));
             DoubleMatrix y = new DoubleMatrix(testData.getOutputs(i));
             sum += (feedForward(x).argmax() == y.argmax()) ? 1 : 0;
         }
         return sum;
    }
    
    private void backProb(DoubleMatrix x, DoubleMatrix y) {
        bDelta = new LinkedList<>();
        wDelta = new LinkedList<>();

        DoubleMatrix activation = x;
        List<DoubleMatrix> activations = new ArrayList<>(NUM_LAYER);
        activations.add(activation);
        List<DoubleMatrix> zVectors = new ArrayList<>(NUM_LAYER - 1);

        // feed forward
        for (int i = 0; i < NUM_LAYER - 1; ++i) {
            DoubleMatrix z = WEIGHTS.get(i).mmul(activation).addColumnVector(BIASES.get(i));
            zVectors.add(z);
            activation = Sigmoid.apply(z.dup());
            activations.add(activation);
        }

        //backward pass
        DoubleMatrix delta = cost.delta(activations.get(NUM_LAYER - 1), y, zVectors.get(NUM_LAYER - 2));
        bDelta.addFirst(delta);
        wDelta.addFirst(delta.mmul(activations.get(NUM_LAYER - 2).transpose()));

        for (int i = NUM_LAYER - 2; i > 0; --i) {
            delta = WEIGHTS.get(i).transpose().mmul(delta).mul(Sigmoid.gradient(zVectors.get(i - 1)));
            bDelta.addFirst(delta);
            wDelta.addFirst(delta.mmul(activations.get(i - 1).transpose()));
        }
    }

    private DoubleMatrix costDerivative(DoubleMatrix a, DoubleMatrix y) {
        return a.sub(y);
    }

    private void updateMiniBatch(TrainingData miniBatch, double eta) {
        List<DoubleMatrix> nablaB = new ArrayList<>(NUM_LAYER - 1);
        List<DoubleMatrix> nablaW = new ArrayList<>(NUM_LAYER - 1);

        for (int i = 0; i < NUM_LAYER - 1; ++i) {
            nablaB.add(DoubleMatrix.zeros(BIASES.get(i).rows, BIASES.get(i).columns));
            nablaW.add(DoubleMatrix.zeros(WEIGHTS.get(i).rows, WEIGHTS.get(i).columns));
        }

        for (int i = 0; i < miniBatch.size(); ++i) {
            DoubleMatrix x = new DoubleMatrix(miniBatch.getInputs(i));
            DoubleMatrix y = new DoubleMatrix(miniBatch.getOutputs(i));
            backProb(x, y);
            for (int j = 0; j < NUM_LAYER - 1; ++j) {
                nablaB.get(j).addi(bDelta.get(j));
                nablaW.get(j).addi(wDelta.get(j));
            }
        }

        double lamda = eta / miniBatch.size();
        for (int i = 0; i < NUM_LAYER - 1; ++i) {
            BIASES.get(i).subi(nablaB.get(i).muli(lamda));
            WEIGHTS.get(i).subi(nablaW.get(i).muli(lamda));
        }
    }

    public void stochasticGradientDescent(DigitDataGenerator trdg, int epochs, 
                int miniBatchSize, double eta, DigitDataGenerator tedg) {
        for (int i = 0; i < epochs; ++i) {
            trdg.shuffle();
            int nMb = trdg.getDataSetSize() / miniBatchSize;
            for (int j = 0; j < nMb; ++j) {
                updateMiniBatch(trdg.getTrainingData(miniBatchSize), eta);
            }
            System.out.println("Epoch : " + (i + 1) + " / " + epochs);
            System.out.println(evaluate(tedg));
        }
    }
}
