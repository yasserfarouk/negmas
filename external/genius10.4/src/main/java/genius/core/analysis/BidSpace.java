package genius.core.analysis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.exceptions.AnalysisException;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;

/**
 * A collection of utilityspaces can be viewed as a space in which a bid is
 * assigned multiple point corresponding to the utility of the bid for different
 * agents. We refer to this space as a BidSpace. This class allows to calculate
 * the properties of this space.
 * 
 * @author Dmytro Tykhonov, Tim Baarslag, Wouter Pasman
 */
public class BidSpace {

	/** Collection of utility spaces constituting the space. */
	private AbstractUtilitySpace[] utilspaces;
	/** Domain of the utility spaces. */
	private Domain domain;
	/** List of all bidpoints in the domain. */
	public ArrayList<BidPoint> bidPoints;

	/** Cached Pareto frontier. */
	List<BidPoint> paretoFrontier = null; // null if not yet computed
	/**
	 * Cached Kalai-Smorodinsky solution. The solution is assumed to be unique.
	 */
	BidPoint kalaiSmorodinsky = null; // null if not yet computed
	/** Cached Nash solution. The solution is assumed to be unique. */
	BidPoint nash = null; // null if not yet computed

	/**
	 * Default constructor used to construct a multidimensional bidding space.
	 * Warning: this call iterates over ALL possible bids.
	 * 
	 * @param utilityspaces
	 *            of which the bidding space consists.
	 * @throws Exception
	 *             is thrown when one of the utility spaces is corrupt.
	 */
	public BidSpace(AbstractUtilitySpace... utilityspaces) throws Exception {
		initializeUtilitySpaces(utilityspaces);
		buildSpace(true);
	}

	/**
	 * Constructor to createFrom a BidSpace given exactly two utility spaces.
	 * The main difference is that if excludeBids is true, then only the bid
	 * points are saved. This has is a good way to save memory. Warning: this
	 * call iterates over ALL possible bids.
	 * 
	 * @param utilityspaceA
	 *            utilityspace of agent A.
	 * @param utilityspaceB
	 *            utilityspace of agent B.
	 * @param excludeBids
	 *            if the real bids should be saved or not.
	 * @throws Exception
	 *             is thrown when one of the utility spaces is corrupt.
	 */
	public BidSpace(AbstractUtilitySpace utilityspaceA, AbstractUtilitySpace utilityspaceB, boolean excludeBids)
			throws Exception {
		AbstractUtilitySpace[] spaces = { utilityspaceA, utilityspaceB };
		initializeUtilitySpaces(spaces);
		buildSpace(excludeBids);
	}

	/**
	 * Constructor which is identical to its three parameter version, except for
	 * the argument skipCheckSpaceB. Independent of the value of this parameter,
	 * this constructor skips the security checks for the second utilityspace.
	 * This is interesting if you use the utility of an opponent model in which
	 * some variables of the utilityspace may not be set. Warning: this call
	 * iterates over ALL possible bids.
	 * 
	 * @param utilityspaceA
	 *            utilityspace of agent A.
	 * @param utilityspaceB
	 *            utilityspace of agent B.
	 * @param excludeBids
	 *            if the real bids should be saved or not.
	 * @param skipCheckSpaceB
	 *            skip security checks for the utilityspace of agent B.
	 * @throws Exception
	 *             if something goes wrong when calculating the utility of a
	 *             bid.
	 */
	public BidSpace(AbstractUtilitySpace utilityspaceA, AbstractUtilitySpace utilityspaceB, boolean excludeBids,
			boolean skipCheckSpaceB) throws Exception {
		if (utilityspaceA == null || utilityspaceB == null)
			throw new NullPointerException("util space is null");
		AbstractUtilitySpace[] spaces = { utilityspaceA, utilityspaceB };
		utilspaces = spaces.clone();
		domain = utilspaces[0].getDomain();
		utilityspaceA.checkReadyForNegotiation(domain);
		buildSpace(excludeBids);
	}

	/**
	 * Initializes the utility spaces by checking if they are valid. This
	 * procedure also clones the spaces such that manipulating them is not
	 * useful for an agent.
	 * 
	 * @param utilityspaces
	 *            to be initialized and validated.
	 * @throws Exception
	 *             if one of the utility spaces is null.
	 */
	private void initializeUtilitySpaces(AbstractUtilitySpace[] utilityspaces) throws Exception {
		utilspaces = utilityspaces.clone();
		for (UtilitySpace utilitySpace : utilityspaces) {
			if (utilitySpace == null)
				throw new NullPointerException("util space is null: " + utilityspaces);
		}
		domain = utilspaces[0].getDomain();
		for (AbstractUtilitySpace space : utilityspaces) {
			space.checkReadyForNegotiation(domain);
		}
	}

	// RA: This method checks whether or not the Pareto file exists
	private boolean checkParetoFileExist(String filePathStr) {

		File f = new File(filePathStr);
		if (f.exists())
			return true;
		else
			return false;
	}

	private void readParetoFromFile(String fileName, boolean isAgentAHasProfile1) {

		this.paretoFrontier = new ArrayList<BidPoint>();
		this.bidPoints = new ArrayList<BidPoint>();
		try {
			FileReader input = new FileReader(fileName);
			BufferedReader bufRead = new BufferedReader(input);
			String line;
			Double[] utility = new Double[2];
			do {
				line = bufRead.readLine();
				if (line != null) {
					int index = line.indexOf(",");
					if (index > 0) {
						if (isAgentAHasProfile1) {
							utility[0] = Double.parseDouble(line.substring(0, line.indexOf(",")));
							utility[1] = Double.parseDouble(line.substring(line.indexOf(",") + 1));
						} else {
							utility[1] = Double.parseDouble(line.substring(0, line.indexOf(",")));
							utility[0] = Double.parseDouble(line.substring(line.indexOf(",") + 1));
						}

						BidPoint bidpt = new BidPoint(null, utility);
						this.paretoFrontier.add(bidpt);
					}
				}

			} while (line != null);

		} catch (IOException e) {
			// If another exception is generated, print a stack trace
			e.printStackTrace();
		}

		System.out.println(this.paretoFrontier);
	}

	/**
	 * Create the space with all bid points from all the
	 * {@link AdditiveUtilitySpace}s.
	 * 
	 * @param excludeBids
	 *            if true do not store the real bids.
	 * @throws exception
	 *             if utility can not be computed for some point.
	 */
	private void buildSpace(boolean excludeBids) throws Exception {

		String fname = utilspaces[0].getFileName();
		if (fname == null) {
			fname = "";
		}

		// RA:
		if (fname.contains("profile-1.xml")) {
			String fileName = fname.replaceAll("profile-1.xml", "pareto.xml");
			if (checkParetoFileExist(fileName)) {
				readParetoFromFile(fileName, true);
				return;
			}
		} else if (fname.contains("profile-2.xml")) {
			String fileName = fname.replaceAll("profile-2.xml", "pareto.xml");
			if (checkParetoFileExist(fileName)) {
				readParetoFromFile(fileName, false);
				return;
			}
		}

		bidPoints = new ArrayList<BidPoint>();
		BidIterator lBidIter = new BidIterator(domain);

		// if low memory mode, do not store the actual. At the time of writing
		// this
		// has no side-effects
		while (lBidIter.hasNext()) {
			Bid bid = lBidIter.next();
			Double[] utils = new Double[utilspaces.length];
			for (int i = 0; i < utilspaces.length; i++) {
				utils[i] = utilspaces[i].getUtility(bid);
			}
			if (excludeBids) {
				bidPoints.add(new BidPoint(null, utils));
			} else {
				bidPoints.add(new BidPoint(bid, utils));
			}
		}
		// System.out.println("Real outcome space:");
		// System.out.println(bidPoints);
	}

	/**
	 * Returns the Pareto fronier. If the Pareto frontier is unknown, then it is
	 * computed using an efficient algorithm. If the utilityspace contains more
	 * than 500000 bids, then a suboptimal algorithm is used.
	 * 
	 * @return The Pareto frontier. The order is ascending utilityA.
	 * @throws Exception
	 *             if the utility of a bid can not be calculated.
	 */
	public List<BidPoint> getParetoFrontier() throws Exception {
		boolean isBidSpaceAvailable = !bidPoints.isEmpty();
		if (paretoFrontier == null) {
			if (isBidSpaceAvailable) {
				paretoFrontier = computeParetoFrontier(bidPoints).getFrontier();
				return paretoFrontier;
			}

			ArrayList<BidPoint> subPareto = new ArrayList<BidPoint>();
			BidIterator lBidIter = new BidIterator(domain);
			ArrayList<BidPoint> tmpBidPoints = new ArrayList<BidPoint>();
			boolean isSplitted = false;
			int count = 0;
			while (lBidIter.hasNext()) {
				Bid bid = lBidIter.next();
				Double[] utils = new Double[utilspaces.length];
				for (int i = 0; i < utilspaces.length; i++)
					utils[i] = utilspaces[i].getUtility(bid);
				tmpBidPoints.add(new BidPoint(bid, utils));
				count++;
				if (count > 500000) {
					subPareto.addAll(computeParetoFrontier(tmpBidPoints).getFrontier());
					tmpBidPoints = new ArrayList<BidPoint>();
					count = 0;
					isSplitted = true;
				}
			}
			// Add the remainder to the sub-Pareto frontier
			if (tmpBidPoints.size() > 0)
				subPareto.addAll(computeParetoFrontier(tmpBidPoints).getFrontier());

			if (isSplitted)
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
	 *            the ArrayList<BidPoint> as computed by BuildSpace and stored
	 *            in bidpoints.
	 * @return the sorted pareto frontier of the bidpoints.
	 * @throws Exception
	 *             if problem occurs
	 */
	private ParetoFrontier computeParetoFrontier(List<BidPoint> points) throws Exception {
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
	 * @throws Exception
	 *             if the utility of a bid cannot be calculated
	 */
	public List<Bid> getParetoFrontierBids() throws Exception {
		ArrayList<Bid> bids = new ArrayList<Bid>();
		List<BidPoint> points = getParetoFrontier();
		for (BidPoint p : points)
			bids.add(p.getBid());
		return bids;
	}

	/**
	 * Calculates Kalai-Smorodinsky optimal outcome. Assumes that Pareto
	 * frontier is already built. Kalai-Smorodinsky is the point on
	 * paretofrontier that has least difference in utilities for A and B.
	 * 
	 * @return the Kalai-Smorodinsky BidPoint.
	 * @throws Exception
	 *             when the Pareto frontier is invalid.
	 */
	public BidPoint getKalaiSmorodinsky() throws Exception {
		if (kalaiSmorodinsky != null)
			return kalaiSmorodinsky;
		if (getParetoFrontier().size() < 1)
			throw new AnalysisException("kalaiSmorodinsky product: Pareto frontier is unavailable.");
		double asymmetry = 2; // every point in space will have lower asymmetry
								// than this.
		for (BidPoint p : paretoFrontier) {
			double asymofp = 0;
			for (int i = 0; i < utilspaces.length; i++) {
				for (int j = i + 1; j < utilspaces.length; j++) {
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

	/**
	 * Calculates the undiscounted Nash optimal outcome. Assumes that Pareto
	 * frontier is already built. Nash is the point on paretofrontier that has
	 * max product of utilities for A and B.
	 * 
	 * @return the Nash BidPoint.
	 * @throws Exception
	 *             when the Pareto frontier is invalid.
	 */
	public BidPoint getNash() throws Exception {
		if (nash != null)
			return nash;
		if (getParetoFrontier().size() < 1)
			throw new AnalysisException("Nash product: Pareto frontier is unavailable.");
		double maxp = -1;
		double[] agentResValue = new double[utilspaces.length];
		for (int i = 0; i < utilspaces.length; i++)
			if (utilspaces[i].getReservationValue() != null)
				agentResValue[i] = utilspaces[i].getReservationValue();
			else
				agentResValue[i] = .0;
		for (BidPoint p : paretoFrontier) {
			double utilofp = 1;
			for (int i = 0; i < utilspaces.length; i++)
				utilofp = utilofp * (p.getUtility(i) - agentResValue[i]);

			if (utilofp > maxp) {
				nash = p;
				maxp = utilofp;
			}
		}
		return nash;
	}

	/**
	 * Returns the nearest Pareto-optimal bid given the opponent's utility
	 * (agent B).
	 * 
	 * @param opponentUtility
	 *            the utility for the opponent.
	 * @return the utility of us on the pareto curve.
	 * @throws Exception
	 *             if getPareto fails or other cases, e.g. paretoFrontier
	 *             contains utilityB = NaN, which may occur if the opponent
	 *             model creating the utility space is corrupt.
	 */
	public double ourUtilityOnPareto(double opponentUtility) throws Exception {

		if (opponentUtility < 0. || opponentUtility > 1.)
			throw new Exception("opponentUtil " + opponentUtility + " is out of [0,1].");
		List<BidPoint> pareto = getParetoFrontier();
		// our utility is along A axis, opp util along B axis.

		// add endpoints to pareto curve such that utilB spans [0,1] entirely
		if (pareto.get(0).getUtility(1) < 1)
			pareto.add(0, new BidPoint(null, new Double[] { 0., 1. }));
		if (pareto.get(pareto.size() - 1).getUtility(1) > 0)
			pareto.add(new BidPoint(null, new Double[] { 1., 0. }));
		if (pareto.size() < 2)
			throw new Exception("Pareto has only 1 point?!" + pareto);
		// pareto is monotonically descending in utilB direction.
		int i = 0;
		while (!(pareto.get(i).getUtility(1) >= opponentUtility && opponentUtility > pareto.get(i + 1).getUtility(1)))
			i++;

		double oppUtil1 = pareto.get(i).getUtility(1); // this is the high value
		double oppUtil2 = pareto.get(i + 1).getUtility(1); // the low value
		double f = (opponentUtility - oppUtil1) / (oppUtil2 - oppUtil1); // f in
																			// [0,1]
																			// is
																			// relative
																			// distance
																			// from
																			// point
																			// i.
		// close to point i means f~0. close to i+1 means f~1
		double lininterpol = (1 - f) * pareto.get(i).getUtility(0) + f * pareto.get(i + 1).getUtility(0);
		return lininterpol;
	}

	/**
	 * @return string representation of the BidSpace, which is basically a long
	 *         list of all bid its bid points.
	 */
	public String toString() {
		return bidPoints.toString();
	}

	/**
	 * Finds the bid with the minimal distance
	 * weightA*DeltaUtilA^2+weightB*DeltaUtilB^2 where DeltaUtilA is the
	 * difference between given utilA and the actual utility of the bid.
	 * 
	 * @param utilA
	 *            the agent-A utility of the point to be found.
	 * @param utilB
	 *            the agent-B utility of the point to be found.
	 * @param weightA
	 *            weight in A direction.
	 * @param weightB
	 *            weight in B direction.
	 * @param excludeList
	 *            Bids to be excluded from the search.
	 * @return best point, or null if none remaining.
	 */
	public BidPoint getNearestBidPoint(double utilA, double utilB, double weightA, double weightB,
			ArrayList<Bid> excludeList) {
		System.out.println("determining nearest bid to " + utilA + "," + utilB);
		System.out.println("excludes=" + excludeList);
		double mindist = 9.; // paretospace distances are always smaller than 2
		BidPoint bestPoint = null;
		double r;
		for (BidPoint p : bidPoints) {
			boolean contains = false;
			for (Bid b : excludeList) {
				if (b.equals(p.getBid())) {
					contains = true;
					break;
				}
			}
			if (contains)
				continue;
			r = weightA * Math.pow((p.getUtility(0) - utilA), 2) + weightB * Math.pow((p.getUtility(1) - utilB), 2);
			if (r < mindist) {
				mindist = r;
				bestPoint = p;
			}
		}
		System.out.println("point found=" + bestPoint.getBid());
		if (excludeList.size() > 1)
			System.out.println("bid equals exclude(1):" + bestPoint.getBid().equals(excludeList.get(1)));
		return bestPoint;
	}

	/**
	 * Method which given a bid point determines the distance to the nearest
	 * Pareto-optimal bid. If the distance is small, than the bid is near
	 * Pareto-optimal.
	 * 
	 * @param bid
	 *            for which the smallest distance to the Pareto frontier is
	 *            found.
	 * @return distance to the nearest Pareto-optimal bid.
	 */
	public double distanceToNearestParetoBid(BidPoint bid) {
		if (paretoFrontier == null) {
			try {
				paretoFrontier = getParetoFrontier();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		double distance = Double.POSITIVE_INFINITY;
		for (BidPoint paretoBid : paretoFrontier) {
			double paretoDistance = bid.getDistance(paretoBid);
			if (paretoDistance < distance) {
				distance = paretoDistance;
			}
		}
		return distance;
	}
}