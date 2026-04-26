package genius.core.analysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.parties.PartyWithUtility;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;
import genius.core.utility.UtilitySpaceTools;

/**
 * Start on analysis of the multi party tournament. Code in this class is mainly
 * adapted from the bilateral analysis which is in the other classes of this
 * package (negotiator.analysis)
 *
 * @author David Festen
 */
public class MultilateralAnalysis {
	/**
	 * Maximum number of bids to analyse
	 */
	public static final int ENUMERATION_CUTOFF = 100000;

	/**
	 * List of all bid points in the domain.
	 */
	private ArrayList<BidPoint> bidPoints;

	/**
	 * Cached Pareto frontier.
	 */
	private List<BidPoint> paretoFrontier = null; // null if not yet computed

	/**
	 * Cached Nash solution. The solution is assumed to be unique.
	 */
	private BidPoint nash = null; // null if not yet computed

	private final BidPoint agreement;

	private final List<? extends PartyWithUtility> parties;

	/**
	 * Collection of utility spaces constituting the space.
	 */
	private List<UtilitySpace> utilitySpaces;

	/**
	 * Domain of the utility spaces.
	 *
	 */
	private Domain domain;

	private Bid agreedBid;

	private final Double endTime;

	/**
	 * @param parties
	 * @param agreedBid
	 *            agreement, or null if there is no agreement.
	 * @param endTime
	 *            the time in range [0,1] at which the negotiation ended where 0
	 *            is the start and 1 the deadline time/round. If null,
	 *            undiscounted utilities will be used.
	 */
	public MultilateralAnalysis(List<? extends PartyWithUtility> parties,
			Bid agreedBid, Double endTime) {
		// System.out.print("Generating analysis... ");

		this.parties = parties;
		this.agreedBid = agreedBid;

		this.endTime = endTime;

		initializeUtilitySpaces(getUtilitySpaces());
		buildSpace(true);

		Double[] utils = new Double[utilitySpaces.size()];
		if (agreedBid == null) {
			for (int i = 0; i < utilitySpaces.size(); i++)
				utils[i] = 0.0;
		} else {
			utils = getUtils(agreedBid);
		}
		agreement = new BidPoint(agreedBid, utils);

		// System.out.println("done");

	}

	public static ArrayList<double[][]> getPartyBidSeries(
			ArrayList<ArrayList<Double[]>> partyUtilityHistoryList) {

		ArrayList<double[][]> bidSeries = new ArrayList<double[][]>();
		double[][] product = new double[2][partyUtilityHistoryList.get(0)
				.size()];
		try {

			for (int i = 0; i < partyUtilityHistoryList.size() - 1; i++) {

				double[][] xPartyUtilities = new double[2][partyUtilityHistoryList
						.get(i).size()];
				int index = 0;

				for (Double[] utilityHistory : partyUtilityHistoryList.get(i)) {

					xPartyUtilities[0][index] = utilityHistory[0];
					xPartyUtilities[1][index] = utilityHistory[1];

					product[0][index] = utilityHistory[0];
					if (i == 0) // for the first agent
						product[1][index] = utilityHistory[1];
					else
						product[1][index] *= utilityHistory[1];
					index++;
				}

				bidSeries.add(xPartyUtilities);
			}
			bidSeries.add(product);

		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

		return bidSeries;
	}

	public List<UtilitySpace> getUtilitySpaces() {
		List<UtilitySpace> spaces = new ArrayList<UtilitySpace>();
		for (PartyWithUtility p : parties) {
			spaces.add(p.getUtilitySpace());
		}
		return spaces;
	}

	/**
	 * Create the space with all bid points from all the
	 * {@link AdditiveUtilitySpace}s.
	 *
	 * @param excludeBids
	 *            if true do not store the real bids.
	 * @throws Exception
	 *             if utility can not be computed for some point.
	 */
	private void buildSpace(boolean excludeBids) {

		bidPoints = new ArrayList<BidPoint>();
		BidIterator lBidIterator = new BidIterator(domain);

		// if low memory mode, do not store the actual. At the time of writing
		// this
		// has no side-effects
		int iterationNumber = 0;
		while (lBidIterator.hasNext()) {
			if (++iterationNumber > ENUMERATION_CUTOFF) {
				// System.out.printf("Could not enumerate complete bid space, "
				// +
				// "enumerated first %d bids... ", ENUMERATION_CUTOFF);
				break;
			}
			Bid bid = lBidIterator.next();
			Double[] utils = getUtils(bid);
			if (excludeBids) {
				bidPoints.add(new BidPoint(null, utils));
			} else {
				bidPoints.add(new BidPoint(bid, utils));
			}
		}
	}

	/**
	 * @return current utility values for all parties as an array
	 */
	private Double[] getUtils(Bid bid) {
		Double[] utils = new Double[utilitySpaces.size()];
		for (int i = 0; i < utilitySpaces.size(); i++) {
			utils[i] = getUtility(bid, utilitySpaces.get(i));
		}
		return utils;
	}

	/**
	 * @param bid
	 * @return utility of a bid, discounted if {@link #endTime} is not null
	 */
	private Double getUtility(Bid bid, UtilitySpace us) {
		return endTime == null ? us.getUtility(bid)
				: us.discount(us.getUtility(bid), endTime);
	}

	/**
	 * Returns the Pareto frontier. If the Pareto frontier is unknown, then it
	 * is computed using an efficient algorithm. If the utility space contains
	 * more than 500000 bids, then a suboptimal algorithm is used.
	 *
	 * @return The Pareto frontier. The order is ascending utilityA.
	 */
	public List<BidPoint> getParetoFrontier() {
		boolean isBidSpaceAvailable = !bidPoints.isEmpty();
		if (paretoFrontier == null) {
			if (isBidSpaceAvailable) {
				paretoFrontier = computeParetoFrontier(bidPoints).getFrontier();
				return paretoFrontier;
			}

			ArrayList<BidPoint> subPareto = new ArrayList<BidPoint>();
			BidIterator lBidIterator = new BidIterator(domain);
			ArrayList<BidPoint> tmpBidPoints = new ArrayList<BidPoint>();
			boolean isSplit = false;
			int count = 0;
			while (lBidIterator.hasNext() && count < ENUMERATION_CUTOFF) {
				Bid bid = lBidIterator.next();
				Double[] utils = getUtils(bid);
				tmpBidPoints.add(new BidPoint(bid, utils));
				count++;
				if (count > 500000) {
					subPareto.addAll(
							computeParetoFrontier(tmpBidPoints).getFrontier());
					tmpBidPoints = new ArrayList<BidPoint>();
					count = 0;
					isSplit = true;
				}
			}
			// Add the remainder to the sub-Pareto frontier
			if (tmpBidPoints.size() > 0)
				subPareto.addAll(
						computeParetoFrontier(tmpBidPoints).getFrontier());

			if (isSplit)
				paretoFrontier = computeParetoFrontier(subPareto).getFrontier(); // merge
			// sub-pareto's
			else
				paretoFrontier = subPareto;
		}
		return paretoFrontier;
	}

	/**
	 * Private because it should be called only with the bids as built by
	 * BuildSpace.
	 *
	 * @param points
	 *            the ArrayList<BidPoint> as computed by
	 *            {@link #buildSpace(boolean)} and stored in bid points.
	 * @return the sorted pareto frontier of the bid points.
	 */
	private ParetoFrontier computeParetoFrontier(List<BidPoint> points) {
		ParetoFrontier frontier = new ParetoFrontier();
		for (BidPoint p : points)
			frontier.mergeIntoFrontier(p);

		frontier.sort();
		return frontier;
	}

	/**
	 * Method which returns a list of the Pareto efficient bids.
	 *
	 * @return Pareto-efficient bids.
	 */
	public List<Bid> getParetoFrontierBids() {
		ArrayList<Bid> bids = new ArrayList<Bid>();
		List<BidPoint> points = getParetoFrontier();
		for (BidPoint p : points)
			bids.add(p.getBid());
		return bids;
	}

	/**
	 * Initializes the utility spaces by checking if they are valid. This
	 * procedure also clones the spaces such that manipulating them is not
	 * useful for an agent.
	 *
	 * @param utilitySpaces
	 *            to be initialized and validated.
	 * @throws NullPointerException
	 *             if one of the utility spaces is null.
	 */
	private void initializeUtilitySpaces(List<UtilitySpace> utilitySpaces) {
		this.utilitySpaces = new ArrayList<UtilitySpace>(utilitySpaces);

		for (UtilitySpace utilitySpace : utilitySpaces)
			if (utilitySpace == null)
				throw new NullPointerException("util space is null");

		domain = this.utilitySpaces.get(0).getDomain();

		for (UtilitySpace space : utilitySpaces) {
			new UtilitySpaceTools(space).checkReadyForNegotiation(domain);
		}
	}

	public double getSocialWelfare() {
		double totalUtility = 0;
		if (agreedBid != null) {
			for (PartyWithUtility agent : parties) {
				totalUtility += getUtility(agreedBid, agent.getUtilitySpace());
			}
		}
		return totalUtility;
	}

	/**
	 * 
	 * @return distance of agreement to nash point
	 */
	public double getDistanceToNash() {
		return agreement.getDistance(getNashPoint());
	}

	/**
	 * 
	 * @return distance of agreement to pareto frontier, or
	 *         {@link Double#POSITIVE_INFINITY} if there is no pareto frontier.
	 */
	public double getDistanceToPareto() {
		double distance = Double.POSITIVE_INFINITY;
		for (BidPoint paretoBid : getParetoFrontier()) {
			double paretoDistance = agreement.getDistance(paretoBid);
			if (paretoDistance < distance) {
				distance = paretoDistance;
			}
		}
		return distance;
	}

	public BidPoint getNashPoint() {
		if (nash != null)
			return nash;
		if (getParetoFrontier().size() < 1) {
			return new BidPoint(null, 0.0, 0.0);
		}
		double maxP = -1;
		double[] agentResValue = new double[utilitySpaces.size()];
		for (int i = 0; i < utilitySpaces.size(); i++)
			if (utilitySpaces.get(i).getReservationValue() != null)
				agentResValue[i] = utilitySpaces.get(i).getReservationValue();
			else
				agentResValue[i] = .0;
		for (BidPoint p : paretoFrontier) {
			double utilOfP = 1;
			for (int i = 0; i < utilitySpaces.size(); i++)
				utilOfP = utilOfP * (p.getUtility(i) - agentResValue[i]);

			if (utilOfP > maxP) {
				nash = p;
				maxP = utilOfP;
			}
		}
		return nash;
	}

	/**
	 * @return a (not necessarily unique) social welfare optimal point. Returns
	 *         null if there are no bids in the space.
	 */

	public BidPoint getSocialwelfarePoint() {
		double max = -1;
		BidPoint maxBid = null;

		for (BidPoint paretoBid : getParetoFrontier()) {
			double welfare = paretoBid.getSocialWelfare();
			if (welfare > max) {
				maxBid = paretoBid;
				max = welfare;
			}
		}
		return maxBid;
	}

	/**
	 * @return kalai-smorodinsky point, or BidPoint(null, 0,0) if utilspace is
	 *         empty.
	 */
	public BidPoint getKalaiPoint() {
		double asymmetry = 2;
		if (getParetoFrontier().size() < 1) {
			return new BidPoint(null, 0.0, 0.0);
		}
		BidPoint kalaiSmorodinsky = null;
		// every point in space will have lower asymmetry than this.
		for (BidPoint p : paretoFrontier) {
			double asymofp = 0;
			for (int i = 0; i < parties.size(); i++) {
				for (int j = i + 1; j < parties.size(); j++) {
					asymofp += Math.abs(p.getUtility(i) - p.getUtility(j));
				}
			}

			if (asymofp < asymmetry) {
				kalaiSmorodinsky = p;
				asymmetry = asymofp;
			}
		}
		return kalaiSmorodinsky;
	}

	public double getOpposition() {
		double opposition = Double.POSITIVE_INFINITY;
		Double[] perfectOutcomeUtils = new Double[this.utilitySpaces.size()];
		Arrays.fill(perfectOutcomeUtils, 1.0);

		BidPoint virtualBestBid = new BidPoint(null, perfectOutcomeUtils);
		for (BidPoint bidPoint : bidPoints) {
			double dist = bidPoint.getDistance(virtualBestBid);
			if (dist < opposition) {
				opposition = dist;
			}
		}

		return opposition;
	}

	/**
	 * 
	 * @return agreement, or null if there is no agreement
	 */
	public Bid getAgreement() {
		return agreedBid;
	}

}
