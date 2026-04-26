package agents.anac.y2018.fullagent;


import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.timeline.Timeline;

public class BidsManager {

	private NegotiationSession negotiationSession;
	
	// the best bid that one of the opponent has bidden in the past
	private Bid maximalOppBid = null;
	private double utilityOfMaximalOppBid = 0;

	// the best bid that one of the opponent has bidden in the past and the second opponent has accepted 
	private Bid maximalBidThatWasAccepted = null;
	private double utilityOfMaximalBidThatWasAccepted = 0;
	
	//the last bid that any opponent has bidden
	private Bid lastOpponentBid = null;

	//time-based protocol or rounds-based protocol
	private Timeline.Type typeOfTimeLine = null;
	double totalTime = 0;
	double currentTimeWhenOurTurnArrived = 0;
	double timeOfTheLastRound = 0;
	
	public BidsManager(NegotiationSession negotiationSession) {
		this.negotiationSession = negotiationSession;
		this.typeOfTimeLine = this.negotiationSession.getTimeline().getType();
		this.totalTime = negotiationSession.getTimeline().getTotalTime();
	}
	
	public Bid getMaximalOppBid() {
		return maximalOppBid;
	}

	public double getUtilityOfMaximalOppBid() {
		return utilityOfMaximalOppBid;
	}

	public Bid getMaximalBidThatWasAccepted() {
		return maximalBidThatWasAccepted;
	}

	public double getUtilityOfMaximalBidThatWasAccepted() {
		return utilityOfMaximalBidThatWasAccepted;
	}
	
	public void reportNewBid(Bid oppBid) {
		//get our undiscounted utility that corresponds to this bid
		this.lastOpponentBid = oppBid;
		double utility = negotiationSession.getUtilitySpace().getUtility(oppBid);
		if (utility > this.utilityOfMaximalOppBid) {
			this.maximalOppBid = oppBid;
			this.utilityOfMaximalOppBid = utility;
		}
	}
	
	public void reportAcceptanceOfBid(Bid oppBid) {
		//if oppBid has bidden by our agent (and not by an opponent) - ignore (i.e. don't save it)
		if (!oppBid.equals(lastOpponentBid)) {
			return;
		}
		//get our undiscounted utility that corresponds to this bid
		double utility = negotiationSession.getUtilitySpace().getUtility(oppBid);
		if (utility > this.utilityOfMaximalBidThatWasAccepted) {
			this.maximalBidThatWasAccepted = oppBid;
			this.utilityOfMaximalBidThatWasAccepted = utility;
		}
	}
	
	public boolean isTheEndOfTimeArriving() {
		double currentTime = negotiationSession.getTimeline().getCurrentTime();
		if (this.typeOfTimeLine == Timeline.Type.Rounds) {
			return (this.totalTime - currentTime <= 2);
		} else if (this.typeOfTimeLine == Timeline.Type.Time) {
			boolean isTheEndArriving = (this.totalTime - currentTime <= 2 * this.timeOfTheLastRound);
			return (isTheEndArriving);
		} else {
			return false;
		}
	}
	
	public void ourTurnHasArrived() {
		if (this.typeOfTimeLine == Timeline.Type.Time) {
			double previousTimeWhenOurTurnArrived = currentTimeWhenOurTurnArrived;
			this.currentTimeWhenOurTurnArrived = this.negotiationSession.getTimeline().getCurrentTime();
			this.timeOfTheLastRound = this.currentTimeWhenOurTurnArrived - previousTimeWhenOurTurnArrived;
		}
	}

}
