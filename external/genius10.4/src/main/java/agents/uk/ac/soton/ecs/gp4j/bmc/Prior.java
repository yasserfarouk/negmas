package agents.uk.ac.soton.ecs.gp4j.bmc;

public interface Prior {
	double[] getSamples();

	double[] getLogSamples();
}
