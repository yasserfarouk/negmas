package agents.anac.y2019.minf.etc;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;

public class TimeDependent_Offering extends OfferingStrategy {
	/** Outcome space */
	private SortedOutcomeSpace outcomespace;
	/** Time of Maximum Utility */
	private double MaxUtilTime;
	/** Time of Concession */
	private double ConcedeTime;
	/** Negotiation Information */
	private NegotiationInfo negotiationInfo;

	public TimeDependent_Offering(NegotiationInfo negoInfo){
		this.negotiationInfo = negoInfo;
	}

	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		super.init(negoSession, parameters);

		this.negotiationSession = negoSession;

		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);

		if (parameters.get("MT") != null)
			this.MaxUtilTime = parameters.get("MT");
		else
			this.MaxUtilTime = 0;

		if (parameters.get("CT") != null)
			this.ConcedeTime = parameters.get("CT");
		else
			this.ConcedeTime = 1.0;

		this.opponentModel = model;

		this.omStrategy = oms;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public BidDetails determineNextBid() {
		double time = negotiationSession.getTime();
		double utilityGoal = negotiationInfo.getRandomThreshold(time);

		if (time < MaxUtilTime) {
			nextBid = negotiationSession.getOutcomeSpace().getMaxBidPossible();
		} else if (time >= ConcedeTime && utilityGoal != 1.0D &&
				negotiationSession.getOpponentBidHistory().getBestBidDetails().getMyUndiscountedUtil()
						>= negotiationSession.getUtilitySpace().getReservationValueUndiscounted()) {
			nextBid = negotiationSession.getOpponentBidHistory().getBestBidDetails();
		} else {
			// if there is no opponent model available
			if (opponentModel instanceof NoModel) {
				nextBid = negotiationSession.getOutcomeSpace().getBidNearUtility(utilityGoal);
			} else {
				nextBid = omStrategy.getBid(outcomespace, utilityGoal);
			}
		}

		return nextBid;
	}

	public NegotiationSession getNegotiationSession() {
		return negotiationSession;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("MT", 0.0, "Max Utility Time"));
		set.add(new BOAparameter("CT", 1.0, "Concede Time"));

		return set;
	}

	@Override
	public String getName() {
		return "TimeDependent Offering";
	}
}
