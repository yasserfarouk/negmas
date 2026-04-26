package negotiator.boaframework.acceptanceconditions.anac2013;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import genius.core.NegotiationResult;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import negotiator.boaframework.offeringstrategy.anac2013.inoxAgent.SaveHelper;

/**
 * This Acceptance Condition will break when the reservation value seems to be a
 * better alternative, and accept when the opponents offer is better than ours
 * or when the opponents offer is higher than a time dependent function.
 * 
 * @author Ruben van Zessen, Mariana Branco
 */
public class AC_InoxAgent extends AcceptanceStrategy {
	/** Discount factor of the domain */
	private double discountFactor;
	/** Median utility in the sorted outcome space */
	private double medianutil;

	/** Time of the previous iteration */
	private double lastTime = 0.0;
	/** List of time differences between iterations */
	private ArrayList<Double> timeList = new ArrayList<Double>();

	/** A check whether the median has been set yet */
	private boolean medianDecided = false;
	/** A check whether the median is the true median or a saved value */
	private boolean realmedian = true;

	/** Previously obtained utility */
	private double prevRes = 0.0;
	/** Number of previous negotiations */
	private int prevNegos = 0;

	/** Estimated number of rounds left in the negotiation */
	private int roundsLeft = Integer.MAX_VALUE;

	/**
	 * Empty constructor.
	 */
	public AC_InoxAgent() {
	}

	/**
	 * Regular constructor.
	 * 
	 * Sets initial values and loads old data if available.
	 */
	public AC_InoxAgent(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel oppModel) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.opponentModel = oppModel;
		discountFactor = negotiationSession.getDiscountFactor();

		Serializable oldData = loadData();
		if (oldData != null) {
			SaveHelper prevData = (SaveHelper) oldData;
			medianutil = prevData.getResult();
			prevRes = prevData.getResult();
			prevNegos = prevData.getNumber();
			realmedian = false;
		}
	}

	/**
	 * Initialization function.
	 * 
	 * Does the same as the regular constructor.
	 */
	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel oppModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.opponentModel = oppModel;
		discountFactor = negotiationSession.getDiscountFactor();
		if (discountFactor == 0.0) {
			discountFactor = 1.0;
		}

		Serializable oldData = loadData();
		if (oldData != null) {
			SaveHelper prevData = (SaveHelper) oldData;
			medianutil = prevData.getResult();
			prevRes = prevData.getResult();
			prevNegos = prevData.getNumber();
			realmedian = false;
		}
	}

	/**
	 * Method which returns the action selected by this acceptance strategy.
	 */
	@Override
	public Actions determineAcceptability() {
		// Determine the median if required
		if (!medianDecided || (roundsLeft <= 4 && !realmedian)) {
			SortedOutcomeSpace outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
			int opplocation = outcomespace.getIndexOfBidNearUtility(
					negotiationSession.getOpponentBidHistory().getFirstBidDetails().getMyUndiscountedUtil());
			List<BidDetails> alloutcomes = outcomespace.getAllOutcomes();
			medianutil = alloutcomes.get((int) Math.floor(((double) opplocation) / 2)).getMyUndiscountedUtil();
			medianDecided = true;
		}

		// Read time and receiveMessage round estimation
		double time = negotiationSession.getTime();
		updateRoundsLeft(time);

		// Determine utility of our worst bid
		double nextMyBidUtil = 1.0;
		if (negotiationSession.getOwnBidHistory().getWorstBidDetails() != null) {
			nextMyBidUtil = negotiationSession
					.getDiscountedUtility(negotiationSession.getOwnBidHistory().getWorstBidDetails().getBid(), time);
		}

		// Determine utility of the opponent bid to be evaluated
		double lastOpponentBidUtil = negotiationSession
				.getDiscountedUtility(negotiationSession.getOpponentBidHistory().getLastBid(), time);

		// Break if we are making an offer that is worse than the reservation
		// value
		if (nextMyBidUtil < negotiationSession.getUtilitySpace().getReservationValueWithDiscount(time)) {
			return Actions.Break;
			// Accept if the opponents offer is better than the one we are about
			// to make (or very close to it in discounted domains)
		} else if (nextMyBidUtil <= lastOpponentBidUtil + 0.05 * (1 - discountFactor)) {
			return Actions.Accept;
			// Accept if the opponents offer is better than the utility from
			// acceptUtil()
		} else if (lastOpponentBidUtil >= acceptUtil(time)) {
			return Actions.Accept;
			// Break if the reservation value appears to be more attractive than
			// continuing negotiation
		} else if (!realmedian && medianutil * Math.pow(discountFactor, time) < negotiationSession.getUtilitySpace()
				.getReservationValueUndiscounted()) {
			return Actions.Break;
		}
		// Reject and make our own offer if none of the above conditions were
		// fulfilled
		return Actions.Reject;
	}

	/**
	 * Method that returns the minimal utility at which we accept an opponents
	 * offer. The function used depends on time and the opponents best
	 * discounted offer.
	 * 
	 * @param t
	 *            Normalized current time in the negotiation
	 */
	public double acceptUtil(double t) {
		double finalVal = negotiationSession.getDiscountedUtility(negotiationSession.getOpponentBidHistory()
				.getBestDiscountedBidDetails(negotiationSession.getUtilitySpace()).getBid(), t);
		finalVal = Math.max(Math.max(finalVal, negotiationSession.getUtilitySpace().getReservationValueWithDiscount(t)),
				medianutil * Math.pow(discountFactor, t));

		if (roundsLeft < 8) {
			return finalVal;
		}
		double startVal = Math.pow(discountFactor, t);
		double power = 27;
		if (discountFactor != 1.0) {
			power = power * 0.3 * discountFactor;
		}
		double ut = startVal - (startVal - (finalVal)) * Math.pow(t, power);
		return ut;
	}

	/**
	 * Method used to estimate the number of rounds that are left by using the
	 * time between iteration for the last 10 rounds.
	 */
	private void updateRoundsLeft(double t) {
		timeList.add(t - lastTime);
		lastTime = t;
		if (timeList.size() >= 10) {
			if (timeList.size() > 10) {
				timeList.remove(0);
			}

			double sum = 0;
			for (int i = 0; i < timeList.size(); i++) {
				sum += timeList.get(i);
			}
			roundsLeft = (int) ((1 - t) * timeList.size() / sum);
		}
	}

	/**
	 * Method used for saving results between sessions.
	 */
	public void endNegotiation(NegotiationResult result) {
		if (result.isAgreement()) {
			double saveRes = (prevRes * prevNegos + result.getMyDiscountedUtility()) / (prevNegos + 1);
			storeData(new SaveHelper(saveRes, prevNegos + 1));
		} else {
			double saveRes = (prevRes * prevNegos + medianutil) / (prevNegos + 1);
			storeData(new SaveHelper(saveRes, prevNegos + 1));
		}
	}

	@Override
	public String getName() {
		return "2013 - InoxAgent";
	}
}