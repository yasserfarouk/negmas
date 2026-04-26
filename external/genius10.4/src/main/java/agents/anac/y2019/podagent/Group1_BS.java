package agents.anac.y2019.podagent;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;
import genius.core.uncertainty.UserModel;

public class Group1_BS extends OfferingStrategy {

	private final int HOW_MANY_BIDS_TO_START_WITH = 5;
	private boolean panicMode = false;
	private boolean extremePanicMode = false;
	private double targetUtility = 1;
	private double undiscountedTargetUtil = 1;
	private double timeLimitForStep = 0.1;
	private List<BidDetails> possibleBids = null;
	private double lastStepTime = 0;
	private NegotiationSession negoSession;
	public OpponentModel opponentModel;
	private SortedOutcomeSpace outcomespace;
	private boolean hardHeaded = false;
	private boolean omHardHeaded = false;
	private Random rand = new Random();
	private Double reservation;


	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		super.init(negoSession, parameters);
		this.negoSession = negoSession;
		this.reservation = negoSession.getUtilitySpace().getReservationValue();
		if(this.reservation == null) {
			System.out.println("No reservation");
		}
		this.opponentModel = model;
		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
		setNewStep();
	}

	@Override
	public BidDetails determineOpeningBid() {
		int bidI = rand.nextInt(HOW_MANY_BIDS_TO_START_WITH);
		bidI = Math.min(bidI, possibleBids.size() - 1);
		return possibleBids.remove(bidI);
	}

	@Override
	public BidDetails determineNextBid() {
		setPanicModeIfNeccesary();

		if (extremePanicMode) {
			return negoSession.getOpponentBidHistory().getBestBidDetails();
		}

		if (possibleBids.isEmpty() || negoSession.getTime() >= timeLimitForStep + lastStepTime || panicMode) {
			setNewStep();
		}
		BidDetails bestBid = null;
		double bestOpponentUtility = 0;
		for (BidDetails bid : possibleBids) {
			double opponentUtil = opponentModel.getBidEvaluation(bid.getBid());
			if (opponentUtil > bestOpponentUtility) {
				bestBid = bid;
				bestOpponentUtility = opponentUtil;
			}
		}

		possibleBids.remove(bestBid);
		return bestBid;
	}

	@Override
	public String getName() {
		return "Group1_BS";
	}

	/**
	 * Calculate concession rate depending on time and friendliness factor
	 * 
	 * @return concession rate between 0 and 1
	 */
	public double getCurrentConcessionRate() {
		double time = negoSession.getTime(); // Between 0 and 1
		double friendlinessFactor = getFriendlinessFactor();
		// Concession function 1 - t^3
		double timeDiscount = (this.undiscountedTargetUtil - (1 - Math.pow(time, 3)));
		// Concession amount is scaled by friendliness factor
		double discount = timeDiscount * friendlinessFactor;
		this.undiscountedTargetUtil = (1 - Math.pow(time, 3));
		double newUtility = this.targetUtility - discount;
		if (newUtility < this.reservation) {
			return this.reservation;
		}
		if (newUtility > 1) {
			return 1;
		} else if (newUtility < 0) {
			return 0;
		}
		return newUtility;
	}

	/**
	 * Reads the friendliness factor from the opponent model
	 * 
	 * @return value between 0 and 1 that indicates the willingness of our opponent
	 *         to cooperate
	 */
	private double getFriendlinessFactor() {
		// If the panic mode is on, ignore the friendliness factor
		if (panicMode) {
			return 1;
		} else if (opponentModel instanceof Group1_OM) {
			omHardHeaded = ((Group1_OM) opponentModel).isHardHeaded();
			return ((Group1_OM) opponentModel).getOpponentSentiment(lastStepTime);
		}
		// Fallback
		else {
			return 0.5;
		}
	}

	/**
	 * Panic mode depending on passed time
	 */
	private void setPanicModeIfNeccesary() {
		// Extreme panic in the last 33% of panic mode
		if (!extremePanicMode && 1 - negoSession.getTime() < 0.05 / 3) {
			extremePanicMode = true;
		}
		// Panic in the last 5% of negotiation time
		if (!panicMode && 1 - negoSession.getTime() < 0.05) {
			panicMode = true;
			hardHeaded = omHardHeaded;
		}
	}

	/**
	 * New step - recalculate concession and range of good bids
	 */
	private void setNewStep() {
		// Make sure the steps don't get too small
		timeLimitForStep = timeLimitForStep > 0.01 ? timeLimitForStep * 0.9 : timeLimitForStep;
		if (!hardHeaded) {
			targetUtility = getCurrentConcessionRate();
		} else {
			// Go back to 1 against hard headed opponents
			targetUtility = 1;
		}
		// Set range of possible bids
		Range range = new Range(targetUtility, 1);
		possibleBids = negoSession.getOutcomeSpace().getBidsinRange(range);
		lastStepTime = this.negotiationSession.getTime();
	}

	/**
	 * Getter for panic mode
	 * 
	 * @return panic mode status
	 */
	public Boolean getPanicMode() {
		return (panicMode);
	}
}
