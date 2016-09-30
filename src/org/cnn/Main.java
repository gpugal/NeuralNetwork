/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.cnn;

import java.io.IOException;
import org.cnn.data.DigitDataGenerator;
import org.cnn.data.DigitImageLoadingService;
import org.cnn.net.Network;
import org.cnn.utility.CrossEntropyCost;

/**
 *
 * @author pugal
 * @date 30 Sep, 2016 - 2:27:13 PM
 */
public class Main {

    public static void main(String[] args) throws IOException {

        DigitImageLoadingService trainingService
                = new DigitImageLoadingService("/home/pugal/work/freelearning/NeuralNetwork/data/train-labels.idx1-ubyte",
                        "/home/pugal/work/freelearning/NeuralNetwork/data/train-images.idx3-ubyte");
        DigitDataGenerator trainingDataGenerator = new DigitDataGenerator(trainingService.loadDigitImages());
        DigitImageLoadingService testingService
                = new DigitImageLoadingService("/home/pugal/work/freelearning/NeuralNetwork/data/t10k-labels.idx1-ubyte",
                        "/home/pugal/work/freelearning/NeuralNetwork/data/t10k-images.idx3-ubyte");
        DigitDataGenerator testingDataGenerator = new DigitDataGenerator(testingService.loadDigitImages());

        Network net = new Network(new int[]{784,100,10},new CrossEntropyCost());
        net.stochasticGradientDescent(trainingDataGenerator, 30, 10, 3.0, testingDataGenerator);
    }
}
