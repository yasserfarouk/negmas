package agents.anac.y2014.BraveCat.OfferingStrategies;

import java.util.HashMap;
import java.util.Random;

import agents.anac.y2014.BraveCat.OpponentModelStrategies.OMStrategy;
import agents.anac.y2014.BraveCat.OpponentModels.NoModel;
import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
import agents.anac.y2014.BraveCat.necessaryClasses.BidGenerator;
import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import agents.anac.y2014.BraveCat.necessaryClasses.Schedular;
import genius.core.bidding.BidDetails;
import genius.core.utility.NonlinearUtilitySpace;

public class BRTOfferingStrategy extends OfferingStrategy {
	private BidGenerator bidGenerator;
	private Random random300;
	private BidDetails LastBestBid = null;
	private double k;
	private double Pmax;
	private double Pmin;
	private double e;// This value should be adjusted by the estimated
						// concession value of the opponent.

	public BRTOfferingStrategy(NegotiationSession negoSession,
			OpponentModel model, OMStrategy oms, double e, double k) {
		random300 = new Random();
		this.e = e;// beta value.
		this.k = k;
		this.negotiationSession = negoSession;
		bidGenerator = new BidGenerator(negotiationSession);

		if (this.negotiationSession.getUtilitySpace() instanceof NonlinearUtilitySpace)
			System.out.println("Nonlinear Utility Space!");
		this.omStrategy = oms;
		try {
			this.Pmax = 1;
			System.out.println("Pmax: " + this.Pmax);
			this.Pmin = this.negotiationSession.getUtilitySpace()
					.getReservationValueUndiscounted();
			System.out.println("Pmin: " + this.Pmin);
			this.schedular = new Schedular(negotiationSession);
		} catch (Exception ex) {
			System.out
					.println("Exception occured when determining Pmax and Pmin!");
		}
	}

	@Override
	public void init(NegotiationSession negoSession, OpponentModel model,
			OMStrategy oms, HashMap<String, Double> parameters)
			throws Exception {
		if (parameters.get("e") != null) {
			this.negotiationSession = negoSession;

			this.e = ((Double) parameters.get("e")).doubleValue();

			if (parameters.get("k") != null)
				this.k = ((Double) parameters.get("k")).doubleValue();
			else {
				this.k = 0.0D;
			}
			if (parameters.get("min") != null)
				this.Pmin = ((Double) parameters.get("min")).doubleValue();
			else {
				this.Pmin = negoSession.getMinBidinDomain()
						.getMyUndiscountedUtil();
			}
			if (parameters.get("max") != null) {
				this.Pmax = ((Double) parameters.get("max")).doubleValue();
			} else {
				BidDetails maxBid = negoSession.getMaxBidinDomain();
				this.Pmax = maxBid.getMyUndiscountedUtil();
			}

			this.opponentModel = model;
			this.omStrategy = oms;
		} else {
			throw new Exception(
					"Constant \"e\" for the concession speed was not set.");
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		BidDetails temp = null;
		try {
			temp = this.bidGenerator.selectBid(Pmax);
			System.out.println("Opening Bid Utility is: "
					+ this.negotiationSession.getUtilitySpace().getUtility(
							temp.getBid()));
		} catch (Exception ex) {
		}
		return temp;
	}

	@Override
	public BidDetails determineNextBid() {
		double time = this.negotiationSession.getTime();

		this.Pmin = this.negotiationSession.getUtilitySpace()
				.getReservationValueWithDiscount(time);

		// If this is the last bid, the agent chooses to send the best bid
		// received from the opponent, so as to avoid a break off,
		// hoping that this bid is still acceptable for the opponent, and it
		// will result in an agreement.
		if (this.schedular.FinalRounds() || this.schedular.LastRound())
			return negotiationSession.getOpponentBidHistory()
					.getBestDiscountedBidDetails(
							negotiationSession.getUtilitySpace());
		double targetUtility = TargetUtilityGenerator(time, this.Pmax,
				this.Pmin, this.e, this.k);
		// System.out.println(targetUtility);
		targetUtility = TargetUtilityRandomizer(targetUtility);
		// System.out.println(targetUtility);
		try {
			targetUtility = BehaviouralTargetUtility(targetUtility);
		} catch (Exception ex) {
			System.out
					.println("Exception occured when calculating behavioral target utility!");
		}
		// System.out.println(targetUtility);
		try {
			if (targetUtility > 1)
				targetUtility = 1;
		} catch (Exception ex) {
			System.out
					.println("Exception occured when bringing the target utility into the range!");
		}

		// System.out.println(targetUtility);

		if ((this.opponentModel instanceof NoModel))
			this.nextBid = this.negotiationSession.getOutcomeSpace()
					.getBidNearUtility(targetUtility);
		else {
			try {
				this.nextBid = this.omStrategy.getBid(this.bidGenerator
						.NBidsNearUtility(targetUtility, 10));
			} catch (Exception ex) {
			}
		}
		return this.nextBid;
	}

	public double BehaviouralTargetUtility(double targetUtility)
			throws Exception {
		if (LastBestBid == null) {
			// System.out.println(this.negotiationSession.getOpponentBidHistory().size());
			LastBestBid = this.negotiationSession.getOpponentBidHistory()
					.getLastBidDetails();
			return targetUtility;
		} else {
			if (this.negotiationSession.getUtilitySpace().getUtility(
					LastBestBid.getBid()) < this.negotiationSession
					.getUtilitySpace().getUtility(
							this.negotiationSession.getOpponentBidHistory()
									.getLastBid())) {
				double NewUtility = targetUtility
						- (this.negotiationSession.getUtilitySpace()
								.getUtility(
										this.negotiationSession
												.getOpponentBidHistory()
												.getLastBid()) - this.negotiationSession
								.getUtilitySpace().getUtility(
										LastBestBid.getBid()));
				LastBestBid = this.negotiationSession.getOpponentBidHistory()
						.getLastBidDetails();
				// System.out.println(NewUtility - targetUtility);
				return NewUtility;
			}
		}
		return targetUtility;
	}

	public double TargetUtilityRandomizer(double targetUtility) {
		Random r = new Random();
		double z = Math.abs(r.nextGaussian());
		return targetUtility
				- ((double) ((1 - targetUtility) * z * (1 - this.negotiationSession
						.getTime())) / 2.58);
	}

	public double TargetUtilityGenerator(double t, double Pmax, double Pmin,
			double e, double k) {
		double temp = Pmin + (Pmax - Pmin) * (1.0D - f(t, e, k));
		return temp;
	}

	public double f(double t, double e, double k) {
		if (e == 0.0D)
			return k;
		double ft = k + (1.0D - k) * Math.pow(t, 1.0D / e);
		return ft;
	}

	public NegotiationSession getNegotiationSession() {
		return this.negotiationSession;
	}

	public double EstimateOpponentConcession() {
		return 1 - ((double) this.negotiationSession.getOpponentBidHistory().numberOfUniqueBidsInBidHistory / this.negotiationSession
				.getOpponentBidHistory().numberOfTotalBidsInBidHistory);
	}

	@Override
	public String GetName() {
		return "BRTOfferingStrategy";
	}
}
