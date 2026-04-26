package negotiator.boaframework.opponentmodel;

import genius.core.Bid;
import genius.core.boaframework.OpponentModel;

/**
 * Opponent model which signals an agent that it should use its default opponent model.
 * 
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M. Jonker.
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * 
 * @author Mark Hendrikx
 */
public class DefaultModel extends OpponentModel {


	public void updateModel(Bid opponentBid, double time) { }
	
	@Override
	public double getBidEvaluation(Bid bid) {
		System.err.println("This model only signals that the default opponent model should be used. " +
				"Check that this model is not applied.");
		return 0.0;
	}
	
	@Override
	public String getName() {
		return "Default Model";
	}
}