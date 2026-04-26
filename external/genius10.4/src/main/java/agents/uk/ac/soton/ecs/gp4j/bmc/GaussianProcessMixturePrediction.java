package agents.uk.ac.soton.ecs.gp4j.bmc;

import agents.Jama.Matrix;
import agents.uk.ac.soton.ecs.gp4j.gp.GaussianProcessPrediction;

public class GaussianProcessMixturePrediction extends GaussianProcessPrediction {

	public GaussianProcessMixturePrediction(Matrix testX, Matrix mean,
			Matrix variance) {
		super(testX, mean, variance);
	}
}
