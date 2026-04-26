package negotiator.boaframework.offeringstrategy.anac2012;

import java.util.HashMap;
import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded.StrategyTypes;
import negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded.TimeDependentFunction;
import negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded.TimeManager;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.opponentmodel.IAMhagglerBayesianModel;

/**
 * The agent which finished third in the ANAC2013 and scored the highest on
 * undiscounted domains. The agent uses a domain analyzer to select a TDT
 * strategy. The parameters of the TDT function are varied based on the discount
 * and the estimated Kalai-point.
 *
 * DEFAULT OM: IAMhagglerBayesianModel with custom parameters
 * 
 * @author Alex Dirzwager, Mark Hendrikx
 */
public class TheNegotiatorReloaded_Offering extends OfferingStrategy {

	private TimeManager timeManager;
	private double kalaiPoint;
	private TimeDependentFunction TDTFunction;
	private StrategyTypes opponentStrategy;

	private enum DiscountTypes {
		High, Medium, Low
	};

	private DiscountTypes discountType;
	private double reservationValue;
	private double maxBidTarget = 1.0;
	private double minBidTarget = 0.7;
	private static int WINDOWS = 60;
	private double discount;

	/**
	 * Default constructor required for the Decoupled Framework.
	 */
	public TheNegotiatorReloaded_Offering() {
	}

	public TheNegotiatorReloaded_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms)
			throws Exception {
		init(negoSession, model, oms, null);
	}

	/**
	 * Init required for the Decoupled Framework.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		if (model instanceof DefaultModel) {
			model = new IAMhagglerBayesianModel();
			HashMap<String, Double> configuration = new HashMap<String, Double>();
			configuration.put("u", 1.0);
			configuration.put("b", 0.3);
			model.init(negoSession, configuration);
			oms.setOpponentModel(model);
		}

		this.opponentModel = model;
		this.omStrategy = oms;

		this.timeManager = new TimeManager(negoSession, opponentModel, oms, WINDOWS);
		this.TDTFunction = new TimeDependentFunction(negoSession, oms, opponentModel);

		discount = negoSession.getDiscountFactor();
		discountType = getDiscountType(discount);

		if (discount > 0.001) {
			minBidTarget = Math.max(0.4, discount * 0.7);
			opponentStrategy = StrategyTypes.Hardliner;
		} else {
			minBidTarget = 0.7;
			opponentStrategy = StrategyTypes.Conceder;
		}

		Double rv = negoSession.getUtilitySpace().getReservationValue();
		reservationValue = 0.0;
		if (rv != null) { // if no reservation value, then it is null
			reservationValue = negoSession.getUtilitySpace().getReservationValue();
		}

		kalaiPoint = 0.7;
	}

	private DiscountTypes getDiscountType(double discount) {
		DiscountTypes type;
		if (discount < 0.0001 || discount >= 0.85) {
			type = DiscountTypes.Low;
		} else if (discount <= 0.4) { // high discount
			type = DiscountTypes.High;
		} else {
			type = DiscountTypes.Medium;
		}
		return type;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return negotiationSession.getMaxBidinDomain();
	}

	@Override
	public BidDetails determineNextBid() {

		if (timeManager.checkEndOfWindow()) {
			kalaiPoint = timeManager.getKalai();
			opponentStrategy = timeManager.getOpponentStrategy();

			if (discount > 0.0001) {
				kalaiPoint = discount * kalaiPoint;
			}

			minBidTarget = Math.max(reservationValue, kalaiPoint);
			maxBidTarget = 1.0;

			if (maxBidTarget < minBidTarget) {
				maxBidTarget = minBidTarget + 0.05;
			}
		}

		switch (opponentStrategy) {
		case Conceder:
			// then we hardline
			if (discountType.equals(DiscountTypes.High)) {
				nextBid = TDTFunction.getNextBid(1.7, 0, minBidTarget, maxBidTarget);
				break;
			} else if (discountType.equals(DiscountTypes.Medium)) {
				nextBid = TDTFunction.getNextBid(0.8, 0, minBidTarget, maxBidTarget);
				break;
			} else {
				nextBid = TDTFunction.getNextBid(0.05, 0, minBidTarget, maxBidTarget);
				break;
			}

		case Hardliner:
			// the we concede
			if (discountType.equals(DiscountTypes.High)) {
				nextBid = TDTFunction.getNextBid(1.9, 0, minBidTarget, maxBidTarget);
				break;
			} else if (discountType.equals(DiscountTypes.Medium)) {
				nextBid = TDTFunction.getNextBid(1.2, 0, minBidTarget, maxBidTarget);
				break;
			} else {
				nextBid = TDTFunction.getNextBid(0.05, 0, minBidTarget, maxBidTarget);
				break;
			}
		}
		return nextBid;
	}

	@Override
	public String getName() {
		return "2012 - TheNegotiatorReloaded";
	}
}