package agents.anac.y2010.Southampton.similarity;

import agents.anac.y2010.Southampton.utils.concession.TimeConcessionFunction;

/**
 * @author Colin Williams
 * A 'similarity' agent with a linear concession strategy.
 * 
 */
public class LinearSimilarityAgent extends VariableConcessionSimilarityAgent {
	public LinearSimilarityAgent() {
		cf = new TimeConcessionFunction(TimeConcessionFunction.Beta.LINEAR, TimeConcessionFunction.BREAKOFF);
	}
}