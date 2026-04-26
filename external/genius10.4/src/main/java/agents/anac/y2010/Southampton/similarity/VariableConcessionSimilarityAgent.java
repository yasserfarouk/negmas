package agents.anac.y2010.Southampton.similarity;

import agents.anac.y2010.Southampton.utils.concession.ConcessionFunction;
import agents.anac.y2010.Southampton.utils.concession.TimeConcessionFunction;
import genius.core.AgentParam;
import genius.core.tournament.VariablesAndValues.AgentParameterVariable;

/**
 * @author Colin Williams
 * A 'similarity' agent with a variable concession strategy.
 * 
 */
public class VariableConcessionSimilarityAgent extends SimilarityAgent {

	protected ConcessionFunction cf;

	/** (non-Javadoc)
	 * @see agents.southampton.similarity.SimilarityAgent#getTargetUtility(double, double)
	 */
	@Override
	protected double getTargetUtility(double myUtility, double oppntUtility) {
		try {
			return getConcession(utilitySpace.getUtility(myPreviousBids.get(0)));
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}
	}

	private double getConcession(double startUtility) {
		if (cf == null)
			cf = new TimeConcessionFunction(parametervalues.get(new AgentParameterVariable(new AgentParam("common", "beta", 0.0, 10.0))).getValue(),
					TimeConcessionFunction.BREAKOFF);
		double currentTime = timeline.getTime() * timeline.getTotalTime() * 1000;
		double totalTime = timeline.getTotalTime() * 1000;
		return cf.getConcession(startUtility, Math.round(currentTime), totalTime);
	}
}
