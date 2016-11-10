package cifo;

public class Statistics {

	public static double sum(double[] sample) {
		double sum = 0.0;
		for (int i = 0; i < sample.length; i++) {
			sum += sample[i];
		}
		return sum;
	}

	public static double mean(double[] sample) {
		return sum(sample) / sample.length;
	}

	public static double standardDeviation(double[] sample) {
		double mean = mean(sample);
		double temp = 0.0;
		for (int i = 0; i < sample.length; i++) {
			temp += Math.pow((sample[i] - mean), 2.0);
		}
		return Math.sqrt(temp / sample.length);
	}

	public static double max(double[] sample) {
		double max = sample[0];
		for (int i = 0; i < sample.length; i++) {
			if (sample[i] > max) {
				max = sample[i];
			}
		}
		return max;
	}

	public static double min(double[] sample) {
		double min = sample[0];
		for (int i = 0; i < sample.length; i++) {
			if (sample[i] < min) {
				min = sample[i];
			}
		}
		return min;
	}
}
