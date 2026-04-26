package negotiator.boaframework.offeringstrategy.anac2013;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import genius.core.NegotiationResult;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import negotiator.boaframework.offeringstrategy.anac2013.inoxAgent.SaveHelper;

/**
 * This class implements a conceding strategy that tries to choose bids that are
 * also good for the opponent.
 * 
 * However, if the opponent shows non-conceding behaviour, this strategy will
 * reset to the best possible bid.
 * 
 * Over time the window that is used to pick bids from is increased, this will
 * happen more quickly on a discounted domain.
 * 
 * @author Ruben van Zessen, Mariana Branco
 */
public class InoxAgent_Offering extends OfferingStrategy {

	/** Initial size of the search window */
	private double startSize;
	/** Final size of the search window */
	private double finSize;
	/** Size of the outcome space */
	private int outcomeSize;
	/** Index of the bid made in the outcome space */
	private int lastBidIndex;
	/** Discount factor of the current domain */
	private double discountFactor;
	/** Median utility in the sorted outcome space */
	private double medianutil;

	/** Previously obtained utility */
	private double prevRes;
	/** Number of previous negotiations */
	private int prevNegos;

	/** A check for whether we made our initial concession move */
	private boolean oneConcession = false;
	/** A check whether the median has been set yet */
	private boolean medianDecided = false;
	/** A check whether the median is the true median or a saved value */
	private boolean realmedian = true;

	/** Time of the previous iteration */
	private double lastTime = 0.0;
	/** List of time differences between iterations */
	private ArrayList<Double> timeList = new ArrayList<Double>();
	/** Estimated number of rounds left in the negotiation */
	private int roundsLeft = Integer.MAX_VALUE;

	/** Outcome space */
	private SortedOutcomeSpace outcomespace;
	/** Best possible bid */
	private BidDetails bestBid;

	/**
	 * Empty constructor.
	 */
	public InoxAgent_Offering() {
	}

	/**
	 * Regular constructor.
	 * 
	 * Sets initial values and loads old data if available.
	 */
	public InoxAgent_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) {
		this.negotiationSession = negoSession;
		discountFactor = negotiationSession.getDiscountFactor();
		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
		List<BidDetails> alloutcomes = outcomespace.getAllOutcomes();
		outcomeSize = alloutcomes.size();
		startSize = 0.01 * outcomeSize;
		finSize = 0.1 * outcomeSize;
		this.opponentModel = model;
		this.omStrategy = oms;

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
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {

		this.negotiationSession = negoSession;
		discountFactor = negotiationSession.getDiscountFactor();
		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
		List<BidDetails> alloutcomes = outcomespace.getAllOutcomes();
		outcomeSize = alloutcomes.size();
		startSize = 0.01 * outcomeSize;
		finSize = 0.1 * outcomeSize;
		this.opponentModel = model;
		this.omStrategy = oms;

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
	 * Initially offer the best possible bid and save it's index in the outcome
	 * space.
	 */
	@Override
	public BidDetails determineOpeningBid() {
		bestBid = negotiationSession.getOutcomeSpace().getMaxBidPossible();
		lastBidIndex = 0;
		return bestBid;
	}

	/**
	 * "Harsh conceder" bidding strategy.
	 * 
	 * This bidding strategy attempts to concede to the opponent by selecting
	 * their best bids in a growing window, starting at it's own best bids. In
	 * the case that the opponent shows non-conceding behaviour, this agent
	 * "resets" it's offer.
	 */
	@Override
	public BidDetails determineNextBid() {
		// Read time and receiveMessage roundsLeft estimate
		double time = negotiationSession.getTime();
		updateRoundsLeft(time);

		// Determine median if is yet to be determined
		if (!medianDecided || (roundsLeft <= 4 && !realmedian)) {
			int opplocation = outcomespace.getIndexOfBidNearUtility(
					negotiationSession.getOpponentBidHistory().getFirstBidDetails().getMyUndiscountedUtil());
			List<BidDetails> alloutcomes = outcomespace.getAllOutcomes();
			medianutil = alloutcomes.get((int) Math.floor(((double) opplocation) / 2)).getMyUndiscountedUtil();
			medianDecided = true;
			realmedian = true;
		}

		int oppHistSize = negotiationSession.getOpponentBidHistory().size();
		BidDetails bestOppBidDetails = negotiationSession.getOpponentBidHistory()
				.getBestDiscountedBidDetails(negotiationSession.getUtilitySpace());

		// If some bid of the opponent was already good for us, make that offer
		// again
		if (bestOppBidDetails.getMyUndiscountedUtil() > negotiationSession.getOwnBidHistory().getWorstBidDetails()
				.getMyUndiscountedUtil()) {
			return bestOppBidDetails;
		}

		// if the opponent shows non-conceding behaviour, reset to our best bid
		if (time < 0.99) {
			if (oppNotConceding(oppHistSize)) {
				// If we havent yet, make our initial single concession
				if (!oneConcession) {
					lastBidIndex = 1;
					oneConcession = true;
					return outcomespace.getAllOutcomes().get(1);
				}
				lastBidIndex = 0;
				return bestBid;
				// early on in the negotiation, concede according to our utility
			} else if (time < 0.05) {
				if (outcomespace.getAllOutcomes().get(lastBidIndex + 1)
						.getMyUndiscountedUtil() > Math.max(
								Math.max(medianutil,
										negotiationSession.getOpponentBidHistory().getBestBidDetails()
												.getMyUndiscountedUtil()),
								negotiationSession.getUtilitySpace().getReservationValueUndiscounted())) {
					lastBidIndex += 1;
					return outcomespace.getAllOutcomes().get(lastBidIndex + 1);
				} else {
					return bestBid;
				}
			}
		}

		// get the outcomes in the current window
		int lowWin = Math.max(lastBidIndex - windowSize(time), 0);
		int upWin = lastBidIndex + windowSize(time) + 1;
		List<BidDetails> bidWindow = outcomespace.getAllOutcomes().subList(lowWin, upWin);

		// get the most fair bid from the window, according to Kalai-Smorodinsky
		BidDetails sendBid = kalaiSmor(bidWindow, lowWin, time);

		// If the opponent has at some point made a better offer than we are
		// about to make, make that offer instead
		if (bestOppBidDetails.getMyUndiscountedUtil() > sendBid.getMyUndiscountedUtil()) {
			sendBid = bestOppBidDetails;
		}

		return sendBid;

	}

	/**
	 * Function that returns true if the opponent is not showing improvement
	 * from the lowest of its last four bids.
	 * 
	 * @param oHS
	 *            The size of the opponents bid history
	 */
	private boolean oppNotConceding(int oHS) {
		double minLastOppUtils = 1.0;
		Iterator<BidDetails> bidIter = negotiationSession.getOpponentBidHistory().getHistory()
				.subList(Math.max(oHS - 4, 0), oHS).iterator();
		while (bidIter.hasNext()) {
			double bid = bidIter.next().getMyUndiscountedUtil();
			if (bid < minLastOppUtils) {
				minLastOppUtils = bid;
			}
		}
		return (minLastOppUtils >= negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil());
	}

	/**
	 * Calculate the size of the window that bids are evaluated in. The window
	 * size increases with time using a 20th order function in an undiscounted
	 * domain, the order decreases when discounts have more influence.
	 */
	private int windowSize(double t) {
		int wSize = (int) Math.ceil(startSize + (finSize - startSize) * Math.pow(t, discountFactor * 20));
		return wSize;
	}

	/**
	 * Determines most fair bid from a list of bids, using the bid that is
	 * closest to a rotating line which converges to the Kalai-Smorodinsky line.
	 * A requirement that the offer needs to be at least 0.12 better for us is
	 * imposed as a safety measure.
	 * 
	 * Also updates the lastBidIndex corresponding to the index of this bid in
	 * the outcome space.
	 */
	private BidDetails kalaiSmor(List<BidDetails> bList, int startIndex, double t) {
		Iterator<BidDetails> it = bList.iterator();
		BidDetails optBid = it.next();
		int currIndex = startIndex;

		// Calculate the slope of the rotating line
		double a;
		if (discountFactor < 0.75) {
			a = Math.max(Math.exp(5 - 5 * (Math.log(discountFactor) / Math.log(0.75)) * t), 1.0);
		} else {
			a = Math.exp(5 - 5 * t);
		}

		// Find the point closest to the line
		while (it.hasNext()) {
			BidDetails itBid = it.next();
			currIndex++;
			double itKalai = Math.min(itBid.getMyUndiscountedUtil(),
					a * opponentModel.getBidEvaluation(itBid.getBid()));
			double optKalai = Math.min(optBid.getMyUndiscountedUtil(),
					a * opponentModel.getBidEvaluation(optBid.getBid()));
			if (itKalai > optKalai
					&& itBid.getMyUndiscountedUtil() >= (opponentModel.getBidEvaluation(itBid.getBid()) + 0.12)) {
				optBid = itBid;
				lastBidIndex = currIndex;
			}
		}

		return optBid;
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
	public void endSession(NegotiationResult result) {
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
		return "2013- INOX";
	}

}