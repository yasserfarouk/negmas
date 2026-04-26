package agents.anac.y2010.Southampton;

import agents.anac.y2010.Southampton.utils.OpponentModel;
import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public class SouthamptonAgentNoExtras implements SouthamptonAgentExtrasInterface {

	@Override
	public void ReceiveMessage(SouthamptonAgent agent, long agentTimeSpent) { }

	@Override
	public void ReceiveMessage() { }

	@Override
	public void chooseAction(SouthamptonAgent agent, long agentTimeSpent) { }

	@Override
	public void chooseAction() { }

	@Override
	public void log(SouthamptonAgent agent, String message) { }

	@Override
	public void postProposeNextBid(SouthamptonAgent agent, Bid myLastBid, AdditiveUtilitySpace utilitySpace, OpponentModel opponentModel, Bid bid) throws Exception { }

	@Override
	public void postReceiveAccept(SouthamptonAgent agent, Bid myLastBid, AdditiveUtilitySpace utilitySpace, OpponentModel opponentModel) { }

	@Override
	public void postSendAccept(SouthamptonAgent agent, Bid myLastBid, AdditiveUtilitySpace utilitySpace, OpponentModel opponentModel, Bid opponentBid) { }

	@Override
	public void preProposeNextBid(SouthamptonAgent agent, Bid myLastBid, AdditiveUtilitySpace utilitySpace, OpponentModel opponentModel, Bid opponentBid) { }

}
