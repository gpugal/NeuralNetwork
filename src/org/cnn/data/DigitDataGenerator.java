/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.cnn.data;

import java.util.*;

public class DigitDataGenerator {

    private List<DigitImage> DigitImageList;
    private int digitCount;
    private int dataSetSize;

    public DigitDataGenerator(List<DigitImage> digitImages) {
        DigitImageList = digitImages;
        dataSetSize = digitImages.size();
    }

    public TrainingData getTrainingData(int miniBatchSize) {
        double[][] inputs = new double[miniBatchSize][DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS];
        double[][] outputs = new double[miniBatchSize][10];        
        for (int i = 0; i < miniBatchSize; i++) {            
            int label = DigitImageList.get(digitCount).getLabel();
            inputs[i] = DigitImageList.get(digitCount).getData();
            outputs[i] = getOutputFor(label);
            digitCount ++;
        }
        return new TrainingData(inputs, outputs);
    }
    
    public void shuffle() {
        digitCount = 0;
        Collections.shuffle(DigitImageList);
    }
    
    public int getDataSetSize() {
        return dataSetSize;
    }

    private double[] getOutputFor(int label) {
        double[] output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        output[label] = 1;
        return output;
    }
}
