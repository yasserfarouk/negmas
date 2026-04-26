package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;

public class DotProductCovarianceFunction extends
		CommutativeCompositeCovarianceFunction {

	public DotProductCovarianceFunction(CovarianceFunction... functions) {
		super(functions);
	}

	@Override
	protected Matrix operation(Matrix matrix1, Matrix matrix2) {
		return matrix1.arrayTimes(matrix2);
	}

}
