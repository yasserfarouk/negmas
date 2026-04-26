package agents.anac.y2019.agentlarry;

import java.util.concurrent.ThreadLocalRandom;

public class LogisticRegression {

    private static final double RATE = 0.5;

    private final double[] weights;

    /**
     * @param sizeOfVector The size of each vector
     */
    public LogisticRegression(int sizeOfVector) {
        this.weights = new double[sizeOfVector];
        for (int i = 0; i < sizeOfVector; i++) {
            this.weights[i] = ThreadLocalRandom.current().nextDouble(-1, 1);
        }
    }

    /**
     * @param number The number to to sigmoid
     * @return The sigmoid of the number
     */
    private static double sigmoid(double number) {
        return 1.0 / (1.0 + Math.exp(-number));
    }

    public void train(Vector vector, double label) {
        double predicted = classify(vector);
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = this.weights[i] + RATE * (label - predicted) * vector.get(i);
        }

    }

    /**
     * @param vector The vector to classify
     * @return The classification of the vector
     */
    public double classify(Vector vector) {
        double sum = 0;
        for (int i = 0; i < this.weights.length ; i++)  {
            sum += this.weights[i] * vector.get(i);
        }
        return sigmoid(sum);
    }
}
