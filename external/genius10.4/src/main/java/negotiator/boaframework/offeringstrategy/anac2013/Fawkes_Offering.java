package negotiator.boaframework.offeringstrategy.anac2013;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;
import negotiator.boaframework.offeringstrategy.anac2013.TheFawkes.Fawkews_Math;
import negotiator.boaframework.omstrategy.TheFawkes_OMS;
import negotiator.boaframework.opponentmodel.TheFawkes_OM;

/**
 * Bidding Strategy
 */
public final class Fawkes_Offering extends OfferingStrategy {
	// Three very important variables -> beta: concession rate formula 7, rho:
	// tolerance threshold formula 1, nu: risk factor formula 7
	private double beta, rho, nu, discountFactor;
	private TheFawkes_OM OM;
	private TheFawkes_OMS OMS;

	@Override
	public void init(NegotiationSession nSession, OpponentModel oppModel, OMStrategy omStrategy,
			Map<String, Double> params) throws Exception {
		super.init(nSession, oppModel, omStrategy, params);
		this.OM = (TheFawkes_OM) oppModel;
		this.OMS = (TheFawkes_OMS) omStrategy;
		this.negotiationSession.setOutcomeSpace(new SortedOutcomeSpace(this.negotiationSession.getUtilitySpace()));

		// The final parameters resulting from tests:
		this.beta = 0.002;
		this.rho = 0.8;
		this.nu = 0.2;

		File biddingParams = new File("g3_biddingparams.txt");
		if (biddingParams.exists() && biddingParams.canRead()) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(biddingParams));
				this.beta = Double.parseDouble(reader.readLine());
				this.rho = Double.parseDouble(reader.readLine());
				this.nu = Double.parseDouble(reader.readLine());
				reader.close();
			} catch (IOException io) {
				io.printStackTrace();
			}
		}

		this.discountFactor = this.negotiationSession.getDiscountFactor();
		this.discountFactor = (this.discountFactor == 0) ? 1 : this.discountFactor; // FIX:
																					// use
																					// discount
																					// factor
																					// 1
																					// when
																					// it
																					// is
																					// given
																					// as
																					// 0
	}

	@Override
	public BidDetails determineOpeningBid() { // Our opening bid is just the
												// maximum possible bid for us
		BidDetails returned = this.negotiationSession.getMaxBidinDomain();
		returned.setTime(this.negotiationSession.getTime());
		return returned;
	}

	@Override
	public BidDetails determineNextBid() {
		BidDetails returned;
		if (this.negotiationSession.getOpponentBidHistory().getHistory().size() > 1) { // If
																						// we
																						// have
																						// anything
																						// to
																						// base
																						// our
																						// OM
																						// on,
																						// use
																						// it
			returned = this.responseMechanism(this.getTargetUtility());
		} else { // Otherwise, just do our opening bid
			returned = this.determineOpeningBid();
		}
		returned.setTime(this.negotiationSession.getTime());
		return returned;
	}

	private BidDetails responseMechanism(double targetUtility) {
		double timeCorrection = 1 - (this.negotiationSession.getTime() / 10.0); // TODO:
																				// do
																				// something
																				// with
																				// discountfactor
																				// here
																				// too?!
		double inertiaCorrection = this.OMS.getSecondBestCount()
				/ (this.negotiationSession.getOwnBidHistory().size() * 10.0); // TODO:
																				// this
																				// is
																				// still
																				// not
																				// ideal...
		double lowerBound = (timeCorrection - inertiaCorrection) * targetUtility;
		// Group3_Agent.debug( "tCor: " + timeCorrection + ", -iCor: " +
		// inertiaCorrection + ", *targetU: " + targetUtility + ", =lB: " +
		// lowerBound );
		List<BidDetails> possibleBids = this.negotiationSession.getOutcomeSpace()
				.getBidsinRange(new Range(lowerBound, Double.MAX_VALUE));
		if (possibleBids.isEmpty()) { // if no solution is found, the latest
										// offer made by the agent is used again
										// in the subsequent round
										// Group3_Agent.debug( "No solution
										// found, using our own last bid again"
										// );
			return this.negotiationSession.getOwnBidHistory().getLastBidDetails();
		} else {
			// Group3_Agent.debug( "Selecting a bid out of " +
			// possibleBids.size() + " bids (" +
			// this.OM.getMaxOpponentBidTimeDiff() + ")" );
			return this.omStrategy.getBid(possibleBids); // selecting a bid in
															// this range is
															// done in the OMS
		}
	}

	/**
	 * Obtain the utility of the next bid we are willing to make, based on our
	 * concession rate and the OM.
	 *
	 * @return Target utility
	 */
	private double getTargetUtility() {
		// Paper step 4 (formulas 7 and 8) is to calculate the reserved utility
		// function and the estimated received utility
		double currentTime = this.negotiationSession.getTime();
		double maxDiff = Double.MIN_VALUE, minDiff = Double.MAX_VALUE, optimisticEstimatedTime = 0,
				optimisticEstimatedUtility = 0, pessimisticEstimatedTime = 0, pessimisticEstimatedUtility = 0;
		for (double time = currentTime; time < Math.min(currentTime + (this.OM.getMaxOpponentBidTimeDiff() * 10),
				1); time += this.OM.getMaxOpponentBidTimeDiff()) { // formulas
																	// 9, 10,
																	// 11, and
																	// 13 are
																	// all
																	// applied
																	// in this
																	// loop:
																	// finding
																	// the
																	// estimated
																	// u and t
																	// (Uhat/That).
																	// Note that
																	// the
																	// time-window
																	// here is
																	// the same
																	// as used
																	// in the AS
			double reservedUtility = this.reservedUtility(time);
			double discountedReservedUtility = this.discountedUtility(reservedUtility, time);
			double estimatedReceivedUtility = this.estimatedReceivedUtility(time);
			double discountedEstimatedReceivedUtility = this.discountedUtility(estimatedReceivedUtility, time);
			double diff = discountedEstimatedReceivedUtility - discountedReservedUtility;
			// Group3_Agent.debug( "(reservedU,estReceivedU)=(" +
			// reservedUtility + "," + estimatedReceivedUtility + ")" + " @" +
			// time );
			if (discountedEstimatedReceivedUtility >= discountedReservedUtility && diff > maxDiff) { // optimistic
																										// result!
				maxDiff = diff; // argmax
				optimisticEstimatedTime = time;
				optimisticEstimatedUtility = estimatedReceivedUtility;
			}
			if (maxDiff == Double.MIN_VALUE) { // once we became optimistic,
												// we'll never be pessimistic
												// this round
				double absoluteDiff = Math.abs(diff);
				if (absoluteDiff < minDiff) {
					minDiff = absoluteDiff; // argmin
					pessimisticEstimatedTime = time;
					pessimisticEstimatedUtility = estimatedReceivedUtility;
				}
			}
		}

		double estimatedTime, estimatedUtility;
		if (maxDiff == Double.MIN_VALUE) { // pessimistic result! use formula 12
			double xsi = (Math.pow(this.rho, -1)
					* this.discountedUtility(pessimisticEstimatedUtility, pessimisticEstimatedTime))
					/ this.discountedUtility(this.reservedUtility(pessimisticEstimatedTime), pessimisticEstimatedTime);
			if (xsi > 1) { // good pessimist :) use the pessimistic values found
				estimatedTime = pessimisticEstimatedTime;
				estimatedUtility = pessimisticEstimatedUtility;
			} else { // panic... no usuable result even when pessimistic (14-1)
				estimatedTime = -1;
				estimatedUtility = -1;
			}
			// Group3_Agent.debug( "[PESSIMIST] X:" + xsi + "(" + ( Math.pow(
			// this.rho, -1 ) * this.discountedUtility(
			// pessimisticEstimatedUtility, pessimisticEstimatedTime ) )
			// + "/" + this.discountedUtility( this.reservedUtility(
			// pessimisticEstimatedTime ), pessimisticEstimatedTime ) + ")" );
		} else { // optimistic result; use the optimistic values found
			estimatedTime = optimisticEstimatedTime;
			estimatedUtility = optimisticEstimatedUtility;

		}
		// Group3_Agent.debug( "T: " + estimatedTime + ", U:" + estimatedUtility
		// );

		double targetUtility;
		if (estimatedUtility == -1) { // still no result (formula 14-1)
			targetUtility = this.reservedUtility(currentTime);
			// Group3_Agent.debug( "[RESERVERDU]=" + targetUtility );
		} else { // formula 14-2
			BidDetails myPreviousBid = this.negotiationSession.getOwnBidHistory().getLastBidDetails();
			double myPreviousUtil = myPreviousBid.getMyUndiscountedUtil(); // targetUtility
																			// is
																			// undiscounted
																			// too
			double factor = (currentTime - estimatedTime) / (myPreviousBid.getTime() - estimatedTime);
			targetUtility = estimatedUtility + (myPreviousUtil - estimatedUtility) * factor;
			// Group3_Agent.debug( "[factor,U]=" + factor + "," + targetUtility
			// );
		}
		return targetUtility;
	}

	/**
	 * Reserved utility function. This function guarantees the minimum utility
	 * at each time step. The calculated utility is dependent on the concession
	 * rate beta.
	 *
	 * @param t
	 *            Normalized time stamp
	 * @return Utility value that is at least the lowest discounted utility
	 *         value at time t. Range: (reservationValue,maxUtility)
	 */
	private double reservedUtility(double time) { // formula 7
		double reservationValue = this.negotiationSession.getUtilitySpace().getReservationValueUndiscounted();
		double myBestBidUtility = this.negotiationSession.getMaxBidinDomain().getMyUndiscountedUtil();
		return reservationValue + (1 - Math.pow(time, 1 / this.beta))
				* ((myBestBidUtility * Math.pow(this.discountFactor, this.nu)) - reservationValue);
	}

	/**
	 * Estimated Received Utility. Expectation of the opponents future
	 * concession. Uses the OM to estimate what the opponent will offer at a
	 * certain time.
	 *
	 * @param time
	 *            Normalized time stamp
	 * @return Utility value that the OM has calculated the opponent will offer
	 *         at time t.
	 */
	private double estimatedReceivedUtility(double time) { // formula 8
		int timeIndex = (int) Math.floor(time * this.OM.timeIndexFactor);
		return this.OM.alpha.evaluate(timeIndex) * (1 + Fawkews_Math.getStandardDeviation(this.OM.ratio));
	}

	/**
	 * The Discounted Utility given a specific time. This is the current utility
	 * multiplied by the discount factor to the power of the time.
	 *
	 * @param utility
	 *            The undiscounted utility.
	 * @param time
	 *            The time you want the discount for.
	 * @return The discounted utility.
	 */
	private double discountedUtility(double utility, double time) {
		return utility * Math.pow(this.discountFactor, time);
	}

	@Override
	public String getName() {
		return "2013 - Fawkes";
	}
}
