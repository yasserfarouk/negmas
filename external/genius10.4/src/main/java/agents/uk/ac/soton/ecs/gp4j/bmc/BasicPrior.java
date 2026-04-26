package agents.uk.ac.soton.ecs.gp4j.bmc;

import agents.org.apache.commons.lang.builder.ToStringBuilder;
import agents.org.apache.commons.math.stat.StatUtils;
import agents.uk.ac.soton.ecs.gp4j.util.ArrayUtils;

public class BasicPrior implements Prior {
	private double logMean;
	private double standardDeviation;
	private double widthScaler = 0.25;
	private int sampleCount;

	public BasicPrior() {

	}

	public BasicPrior(int sampleCount, double mean, double standardDeviation) {
		this(sampleCount, mean, standardDeviation, 0.25);
	}

	public BasicPrior(int sampleCount, double mean, double standardDeviation,
			double widthScaler) {
		super();
		setMean(mean);
		setStandardDeviation(standardDeviation);
		setSampleCount(sampleCount);
		this.widthScaler = widthScaler;

	}

	public void setSampleCount(int sampleCount) {
		this.sampleCount = sampleCount;
	}

	public void setMean(double mean) {
		this.logMean = Math.log(mean);
	}

	public int getSampleCount() {
		return sampleCount;
	}

	public double[] getSamples() {
		double[] samples = getLogSamples();

		for (int i = 0; i < samples.length; i++)
			samples[i] = Math.exp(samples[i]);

		return samples;
	}

	public double getLogMean() {
		return logMean;
	}

	public double getStandardDeviation() {
		return standardDeviation;
	}

	public double[] getLogSamples() {
		double[] logSamples;

		if (sampleCount == 1)
			logSamples = new double[] { logMean };
		else {
			logSamples = ArrayUtils.linspace(logMean - 2 * standardDeviation,
					logMean + 2 * standardDeviation, sampleCount);
		}

		return logSamples;
	}

	public double getWidth() {
		if (getSampleCount() == 1)
			return widthScaler;
		else
			return widthScaler
					* (StatUtils.max(getLogSamples()) - StatUtils
							.min(getLogSamples())) / (sampleCount - 1);
	}

	@Override
	public String toString() {
		return ToStringBuilder.reflectionToString(this);
	}

	public void setStandardDeviation(double standardDeviation) {
		this.standardDeviation = standardDeviation;
	}

	public static void main(String[] args) {
		System.out.println(ArrayUtils.toString(new BasicPrior(11, 3, .4)
				.getSamples()));
	}
}
