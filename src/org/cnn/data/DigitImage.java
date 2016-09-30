/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.cnn.data;

public class DigitImage {

    private final int LABEL;
    private final double[] DATA;
    private final double OFFSET = 1.0 / 255;

    public DigitImage(int label, byte[] data) {
        this.LABEL = label;
        this.DATA = new double[data.length];
        for (int i = 0; i < this.DATA.length; i++) {
            this.DATA[i] = (data[i] & 0xFF) * OFFSET;
        }
    }

    public int getLabel() {
        return LABEL;
    }

    public double[] getData() {
        return DATA;
    }
}
