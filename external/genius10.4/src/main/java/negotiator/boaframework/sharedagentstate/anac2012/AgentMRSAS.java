package negotiator.boaframework.sharedagentstate.anac2012;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC2012 AgentMR.
 * The code was taken from the ANAC2012 Agent MR and adapted to work within the BOA framework.
 * 
 * @author Alex Dirkzwager
 */
public class AgentMRSAS extends SharedAgentState {
	
	private NegotiationSession negotiationSession;
	private double sigmoidGain;
	private double sigmoidX;
	private double reservation = 0.0;
	private double alpha;
	private double percent;
	private double p = 0.90;
	private static double minimumBidUtility;
	private static double minimumOffereDutil;
	private ArrayList<Bid> bidRunk = new ArrayList<Bid>();
	double firstOffereUtility; 
	boolean calulatedFirstBidUtility = false;

	public AgentMRSAS (NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "AgentMR";	
	}

	public ArrayList<Bid> getBidRunk() {
		return bidRunk;
	}

	public void setBidRunk(ArrayList<Bid> bidRunk) {
		this.bidRunk = bidRunk;
	}
	
	public void updateMinimumBidUtility(double time) {
		alpha = (1.0 - getFirstOffereUtility()) * percent; 
		//System.out.println("Decoupled percent: " + percent);
		//System.out.println("Decoupled alpha: " + alpha);
		//System.out.println("Decoupled sigmoidGain: " + sigmoidGain);


		double mbuInfimum = getFirstOffereUtility() + alpha;


		if (mbuInfimum >= 1.0) {
			mbuInfimum = 0.999; 
		} else if (mbuInfimum <= 0.70) {
			mbuInfimum = 0.70; 
		}
		sigmoidX = 1 - ((1 / sigmoidGain) * Math.log(mbuInfimum / (1 - mbuInfimum)));

		minimumBidUtility = 1 - (1 / (1 + Math.exp(sigmoidGain
				* (time - sigmoidX)))); 
		//System.out.println("Decoupled sigmoidX: " + sigmoidX);
		//System.out.println("Decoupled sigmoidGain: " + sigmoidGain);

		if (minimumBidUtility < reservation) { 
			minimumBidUtility = reservation;
		}

		minimumOffereDutil =  minimumBidUtility * p;

	}
	
	public void calculateFirstOffereUtility(){
		if (negotiationSession.getUtilitySpace().isDiscounted()) {
			firstOffereUtility = negotiationSession.getOpponentBidHistory().getFirstBidDetails().getMyUndiscountedUtil()
					* (1 / Math.pow(negotiationSession.getUtilitySpace().getDiscountFactor(),
							negotiationSession.getOpponentBidHistory().getFirstBidDetails().getTime()));
		} else {
			firstOffereUtility = negotiationSession.getOpponentBidHistory().getFirstBidDetails().getMyUndiscountedUtil();
		}
	}	
	

	public double getFirstOffereUtility(){
		return firstOffereUtility;
	}
	
	public void setFirstOffereUtility(double value){
		firstOffereUtility = value;
	}
	
	public double getSigmoidGain() {
		return sigmoidGain;
	}

	public void setSigmoidGain(double sigmoidGain) {
		this.sigmoidGain = sigmoidGain;
	}

	public NegotiationSession getNegotiationSession() {
		return negotiationSession;
	}

	public void setNegotiationSession(NegotiationSession negotiationSession) {
		this.negotiationSession = negotiationSession;
	}

	public double getSigmoidX() {
		return sigmoidX;
	}

	public void setSigmoidX(double sigmoidX) {
		this.sigmoidX = sigmoidX;
	}

	public double getReservation() {
		return reservation;
	}

	public void setReservation(double reservation) {
		this.reservation = reservation;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double getP() {
		return p;
	}

	public void setP(double p) {
		this.p = p;
	}

	public double getMinimumBidUtility() {
		return minimumBidUtility;
	}

	public double getMinimumOffereDutil() {
		return minimumOffereDutil;
	}

	public double getPercent() {
		return percent;
	}

	public void setPercent(double percent) {
		this.percent = percent;
	}
}
