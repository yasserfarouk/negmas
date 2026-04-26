package agents.anac.y2015.agentBuyogV2;

import agents.anac.y2015.agentBuyogV2.flanagan.analysis.RegressionFunction;


public class LearningFunction implements RegressionFunction{

	private double y, initialUtility;
	
	public LearningFunction(double initialUtility){
		this.initialUtility = initialUtility;
	}
	
	@Override
	public double function(double[] param, double[] x) {
		y = initialUtility + Math.pow(Math.E, param[0])*Math.pow(x[0], param[1]);
		return y;
	}

}
