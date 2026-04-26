package agents;

import java.util.ArrayList;

import agents.bayesianopponentmodel.BayesianOpponentModel;
import agents.bayesianopponentmodel.OpponentModel;
import agents.bayesianopponentmodel.OpponentModelUtilSpace;
import genius.core.Agent;
import genius.core.AgentParam;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.analysis.BidPoint;
import genius.core.analysis.BidSpace;
import genius.core.tournament.VariablesAndValues.AgentParameterVariable;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Wrapper for opponentmodelspace, so that it is a neat utilityspace that we can
 * give to the bidspace.
 * 
 * @author wouter
 * 
 */

public class BayesianAgentForAuction extends Agent {

	protected Action messageOpponent;
	protected Bid myLastBid = null;
	protected Action myLastAction = null;
	protected Bid fOpponentPreviousBid = null;

	protected enum ROLE {
		CENTER, PROVIDER, IRRELEVANT
	};

	public enum PHASE {
		FIRST_PHASE, SECOND_PHASE
	};

	protected enum ACTIONTYPE {
		START, OFFER, ACCEPT, BREAKOFF
	};

	private enum STRATEGY {
		SMART, SERIAL, RESPONSIVE, RANDOM, TIT_FOR_TAT, AUCTION
	};

	private STRATEGY fStrategy = STRATEGY.AUCTION;
	private boolean fMarketPreassure = false;
	protected int fSmartSteps;
	protected OpponentModel fOpponentModel;
	protected OpponentModel[] fOpponentModels = null;
	protected double CONCESSIONFACTOR = 0.04;
	private static final double ALLOWED_UTILITY_DEVIATION = 0.01;
	protected static final int NUMBER_OF_SMART_STEPS = 0;
	protected ArrayList<Bid> myPreviousBids;
	private boolean fDebug = false;
	protected PHASE fPhase = null;
	protected ROLE fRole;

	// Class constructor
	public BayesianAgentForAuction() {
		super();

	}

	@Override
	public String getVersion() {
		return "2.0";
	}

	/** Dummy variables, for testing only. W.Pasman 19aug08 */
	public static ArrayList<AgentParam> getParameters() {
		ArrayList<AgentParam> parameters = new ArrayList<AgentParam>();
		parameters.add(new AgentParam(BayesianAgentForAuction.class.getName(),
				"reservation", 0., 1.));
		parameters.add(new AgentParam(BayesianAgentForAuction.class.getName(),
				"starting_utility", 0., 1.));
		parameters.add(new AgentParam(BayesianAgentForAuction.class.getName(),
				"role", -1., 3.));
		parameters.add(new AgentParam(BayesianAgentForAuction.class.getName(),
				"phase", -1., 1.));
		parameters.add(new AgentParam(BayesianAgentForAuction.class.getName(),
				"opponent", -1., 1.));
		return parameters;
	}

	public void init() {
		messageOpponent = null;
		myLastAction = null;
		fSmartSteps = 0;
		myPreviousBids = new ArrayList<Bid>();
		fOpponentPreviousBid = null;
		if ((getParameterValues().isEmpty())
				|| (getParameterValues().get(
						new AgentParameterVariable(new AgentParam(
								BayesianAgentForAuction.class.getName(),
								"phase", new Double(-1.), new Double(1.))))
						.getValue() < 0))
			fPhase = PHASE.FIRST_PHASE;
		else
			fPhase = PHASE.SECOND_PHASE;
		if (getParameterValues().isEmpty())
			fRole = ROLE.IRRELEVANT;
		else if ((getParameterValues().get(
				new AgentParameterVariable(new AgentParam(
						BayesianAgentForAuction.class.getName(), "role", -1.,
						3.))).getValue() < 0))
			fRole = ROLE.PROVIDER;
		else if (getParameterValues().get(
				new AgentParameterVariable(new AgentParam(
						BayesianAgentForAuction.class.getName(), "role", -1.,
						3.))).getValue() < 2) {
			fRole = ROLE.CENTER;
			CONCESSIONFACTOR = 0.06;
		} else
			fRole = ROLE.IRRELEVANT;
		if ((fPhase == PHASE.FIRST_PHASE) && (fOpponentModels == null))
			prepareOpponentModel();
		if (fRole == ROLE.CENTER) {
			int index = Double.valueOf(
					getParameterValues().get(
							new AgentParameterVariable(new AgentParam(
									BayesianAgentForAuction.class.getName(),
									"opponent", -1., 1.))).getValue())
					.intValue();
			fOpponentModel = fOpponentModels[index];
		}

	}

	protected void prepareOpponentModel() {
		fOpponentModels = new BayesianOpponentModel[2];
		fOpponentModels[0] = new BayesianOpponentModel(
				(AdditiveUtilitySpace) utilitySpace);
		fOpponentModels[1] = new BayesianOpponentModel(
				(AdditiveUtilitySpace) utilitySpace);
		fOpponentModel = fOpponentModels[0];
	}

	// Class methods
	public void ReceiveMessage(Action opponentAction) {
		messageOpponent = opponentAction;
	}

	protected Action proposeInitialBid() throws Exception {
		Bid lBid = null;
		switch (fRole) {
		case CENTER:
			switch (fPhase) {
			case FIRST_PHASE:
				lBid = getMaxUtilityBid();
				break;
			case SECOND_PHASE:
				double lSecondBest = getParameterValues().get(
						new AgentParameterVariable(new AgentParam(
								BayesianAgentForAuction.class.getName(),
								"reservation", 0., 1.))).getValue();
				lBid = getTradeOff(lSecondBest);
				break;
			}
			break;
		case PROVIDER:
			switch (fPhase) {
			case FIRST_PHASE:
				double lReservationValue = utilitySpace.getReservationValue();// parametervalues.get("reservation");
				lBid = getTradeOff(lReservationValue);
				break;
			case SECOND_PHASE:
				lBid = getMaxUtilityBid();
				break;
			}
			break;
		case IRRELEVANT:
			switch (fPhase) {
			case FIRST_PHASE:
				// if(fRole=ROLE.)
				lBid = getMaxUtilityBid();
				break;
			case SECOND_PHASE:
				double lSecondBest = getParameterValues().get(
						new AgentParameterVariable(new AgentParam(
								BayesianAgentForAuction.class.getName(),
								"starting_utility", 0., 1.))).getValue();
				lBid = getTradeOff(lSecondBest);
				break;
			}
			break;
		}
		// Return (one of the) possible bid(s) with maximal utility.
		fSmartSteps = NUMBER_OF_SMART_STEPS;
		myLastBid = lBid;
		return new Offer(getAgentID(), lBid);
	}

	/**
	 * 
	 * @param pOppntBid
	 * @return a counterbid that has max util for us and an opponent utility
	 *         that is equal to 1-estimated utility of opponent's last bid. Or,
	 *         if that bid was done already before, another bid that has same
	 *         utility in our space as that counterbid.
	 * @throws Exception
	 */
	protected Bid getNextBidAuction(Bid pOppntBid) throws Exception {
		if (pOppntBid == null)
			throw new NullPointerException("pOpptBid=null");
		if (myLastBid == null)
			throw new Exception("myLastBid==null");
		log("Get next bid ...");
		Bid lBid = null;

		switch (fRole) {
		case CENTER:
			switch (fPhase) {
			case FIRST_PHASE:
				lBid = getNextBidSmart(pOppntBid);
				break;
			case SECOND_PHASE:
				double lSecondBest = getParameterValues().get(
						new AgentParameterVariable(new AgentParam(
								BayesianAgentForAuction.class.getName(),
								"reservation", 0., 1.))).getValue();
				lBid = getTradeOff(lSecondBest);
				break;
			}
			break;
		case PROVIDER:
			switch (fPhase) {
			case FIRST_PHASE:
				double lReservationValue = utilitySpace.getReservationValue();// parametervalues.get("reservation");
				lBid = getTradeOff(lReservationValue);
				break;
			case SECOND_PHASE:
				lBid = getNextBidSmart(pOppntBid);
				break;
			}
			break;
		case IRRELEVANT:
			switch (fPhase) {
			case FIRST_PHASE:
				lBid = getNextBidSmart(pOppntBid);
				break;
			case SECOND_PHASE:
				lBid = getNextBidSmart(pOppntBid);
				break;
			}
			break;

		}
		return lBid;
	}

	/**
	 * 
	 * @param pOppntBid
	 * @return a counterbid that has max util for us and an opponent utility
	 *         that is equal to 1-estimated utility of opponent's last bid. Or,
	 *         if that bid was done already before, another bid that has same
	 *         utility in our space as that counterbid.
	 * @throws Exception
	 */
	private Bid getNextBid(Bid pOppntBid) throws Exception {
		if (pOppntBid == null)
			throw new NullPointerException("pOpptBid=null");
		if (myLastBid == null)
			throw new Exception("myLastBid==null");
		log("Get next bid ...");

		BidSpace bs = new BidSpace((AdditiveUtilitySpace) utilitySpace,
				new OpponentModelUtilSpace(fOpponentModel), true, true);
		// System.out.println("Bidspace:\n"+bs);

		// compute opponent's concession
		double opponentConcession = 0.;
		if (fOpponentPreviousBid == null)
			opponentConcession = 0;
		else {
			double opponentUtil = fOpponentModel
					.getNormalizedUtility(pOppntBid);
			double opponentFirstBidUtil = fOpponentModel
					.getNormalizedUtility(fOpponentModel.fBiddingHistory.get(0));
			opponentConcession = opponentUtil - opponentFirstBidUtil;
		}
		log("opponent Concession:" + opponentConcession);

		// determine our bid point
		double OurFirstBidOppUtil = fOpponentModel
				.getNormalizedUtility(myPreviousBids.get(0));
		double OurTargetBidOppUtil = OurFirstBidOppUtil - opponentConcession;
		if (OurTargetBidOppUtil > 1)
			OurTargetBidOppUtil = 1.;
		if (OurTargetBidOppUtil < OurFirstBidOppUtil)
			OurTargetBidOppUtil = OurFirstBidOppUtil;
		log("our target opponent utility=" + OurTargetBidOppUtil);

		// find the target on the pareto curve
		double targetUtil = bs.ourUtilityOnPareto(OurTargetBidOppUtil);

		BidPoint bp = bs.getNearestBidPoint(targetUtil, OurTargetBidOppUtil,
				.5, 1, myPreviousBids);
		log("found bid " + bp);
		return bp.getBid();
	}

	/**
	 * get a new bid (not done before) that has ourUtility for us.
	 * 
	 * @param ourUtility
	 * @return the bid with max opponent utility that is close to ourUtility. or
	 *         null if there is no such bid.
	 */
	Bid getNewBidWithUtil(double ourUtility, BidSpace bs) {
		BidPoint bestbid = null;
		double bestbidutil = 0;
		for (BidPoint p : bs.bidPoints) {
			if (Math.abs(ourUtility - p.getUtilityA()) < ALLOWED_UTILITY_DEVIATION
					&& p.getUtilityB() > bestbidutil
					&& !myPreviousBids.contains(p.getBid())) {
				bestbid = p;
				bestbidutil = p.getUtilityB();
			}
		}
		if (bestbid == null)
			return null;
		return bestbid.getBid();
	}

	/**
	 * Wouter: Try to find a bid that has same utility for ourself but max
	 * utility for opponent.
	 * 
	 * @author Dmytro
	 * @param pBid
	 * @return
	 * @throws Exception
	 */
	protected Bid getSmartBid(Bid pBid) throws Exception {
		Bid lBid = null;
		double lExpectedUtility = -1;
		double lUtility = utilitySpace.getUtility(pBid);
		BidIterator lIter = new BidIterator(utilitySpace.getDomain());
		// int i=1;
		while (lIter.hasNext()) {
			Bid tmpBid = lIter.next();
			// System.out.println(tmpBid);
			// System.out.println(String.valueOf(i++));
			if (Math.abs(utilitySpace.getUtility(tmpBid) - lUtility) < ALLOWED_UTILITY_DEVIATION) {
				// double lTmpSim = fSimilarity.getSimilarity(tmpBid,
				// pOppntBid);
				double lTmpExpecteUtility = fOpponentModel
						.getNormalizedUtility(tmpBid);
				if (lTmpExpecteUtility > lExpectedUtility) {
					lExpectedUtility = lTmpExpecteUtility;
					lBid = tmpBid;
				}
			}
		}
		return lBid;
		// check if really found a better bid. if not return null
		/*
		 * if(fOpponentModel.getNormalizedUtility(lBid)>(fOpponentModel.
		 * getNormalizedUtility(pBid)+0.04)) return lBid; else return null;
		 */
	}

	protected Bid getNextBidSmart(Bid pOppntBid) throws Exception {
		double lMyUtility, lOppntUtility, lTargetUtility;
		// Both parties have made an initial bid. Compute associated utilities
		// from my point of view.
		lMyUtility = utilitySpace.getUtility(myLastBid);
		lOppntUtility = utilitySpace.getUtility(pOppntBid);
		if (fSmartSteps >= NUMBER_OF_SMART_STEPS) {
			lTargetUtility = getTargetUtility(lMyUtility, lOppntUtility);
			fSmartSteps = 0;
		} else {
			lTargetUtility = lMyUtility;
			fSmartSteps++;
		}
		if (lTargetUtility < utilitySpace.getReservationValue())
			return null;
		else
			return getTradeOff(lTargetUtility);
	}

	protected Bid getTradeOff(double pUtility) throws Exception {
		if (pUtility < utilitySpace.getReservationValue())
			return null;
		Bid lBid = null;
		double lExpectedUtility = -100;
		BidIterator lIter = new BidIterator(utilitySpace.getDomain());
		// int i=1;
		while (lIter.hasNext()) {
			Bid tmpBid = lIter.next();
			// System.out.println(tmpBid);
			// System.out.println(String.valueOf(i++));
			if (fMarketPreassure)
				if (fNegotiation.getOpponentUtility(this, tmpBid) < 0.3)
					continue;
			if (Math.abs(utilitySpace.getUtility(tmpBid) - pUtility) < ALLOWED_UTILITY_DEVIATION) {
				// double lTmpSim = fSimilarity.getSimilarity(tmpBid,
				// pOppntBid);
				double lTmpExpecteUtility = fNegotiation.getOpponentUtility(
						this, tmpBid); // fOpponentModel.getExpectedUtility(tmpBid);
				if (lTmpExpecteUtility > lExpectedUtility) {
					lExpectedUtility = lTmpExpecteUtility;
					lBid = tmpBid;
				}
			}
		} // while
		return lBid;
	}

	protected Bid proposeNextBid(Bid pOppntBid) throws Exception {
		Bid lBid = null;
		switch (fStrategy) {
		case TIT_FOR_TAT:
			lBid = getNextBid(pOppntBid);
			break;
		case SMART:
			lBid = getNextBidSmart(pOppntBid);
			break;
		case AUCTION:
			lBid = getNextBidAuction(pOppntBid);
			break;
		default:
			throw new Exception("unknown strategy " + fStrategy);
		}
		myLastBid = lBid;
		return lBid;
	}

	public double getOpponentUtility(Bid bid) throws Exception {
		return fOpponentModel.getExpectedUtility(bid);
	}

	public Action chooseAction() {
		Action lAction = null;
		ACTIONTYPE lActionType;
		Bid lOppntBid = null;

		try {
			lActionType = getActionType(messageOpponent);
			switch (lActionType) {
			case OFFER: // Offer received from opponent
				lOppntBid = ((Offer) messageOpponent).getBid();
				// if (fOpponentModel.haveSeenBefore(lOppntBid)) {
				// lAction=myLastAction; break; }
				// double lDistance = calculateEuclideanDistanceUtilitySpace();
				// if(myLastAction==null) dumpDistancesToLog(0);
				System.out.print("Updating beliefs ...");
				if (myPreviousBids.size() < 8)
					fOpponentModel.updateBeliefs(lOppntBid);
				// dumpDistancesToLog(fRound++);
				System.out.println("Done!");
				if (myLastAction == null)
					// Other agent started, lets propose my initial bid.
					lAction = proposeInitialBid();
				else {
					double offeredutil = utilitySpace.getUtility(lOppntBid);
					double time = timeline.getTime();
					double P = Paccept(offeredutil, time);
					log("time=" + time + " offeredutil=" + offeredutil
							+ " accept probability P=" + P);
					if (utilitySpace.getUtility(lOppntBid) * 1.05 >= utilitySpace
							.getUtility(myLastBid)
					/* || .05*P>Math.random() */) {
						// Opponent bids equally, or outbids my previous bid, so
						// lets accept
						lAction = new Accept(getAgentID(), lOppntBid);
						log("randomly accepted");
					} else {
						Bid lnextBid = proposeNextBid(lOppntBid);
						if (lnextBid == null) {
							lAction = new EndNegotiation(getAgentID());
						} else {

							lAction = new Offer(getAgentID(), lnextBid);

							// Propose counteroffer. Get next bid.
							// Check if utility of the new bid is lower than
							// utility of the opponent's last bid
							// if yes then accept last bid of the opponent.
							if (utilitySpace.getUtility(lOppntBid) * 1.05 >= utilitySpace
									.getUtility(lnextBid)) {
								// Opponent bids equally, or outbids my previous
								// bid, so lets accept
								lAction = new Accept(getAgentID(), lOppntBid);
								log("opponent's bid higher than util of my last bid! accepted");
							}
						}

					}
					// remember current bid of the opponent as its previous bid
					fOpponentPreviousBid = lOppntBid;
				}
				break;
			case ACCEPT:
			case BREAKOFF:
				// nothing left to do. Negotiation ended, which should be
				// checked by
				// Negotiator...
				break;
			default:
				// I am starting, but not sure whether Negotiator checks this,
				// so
				// lets check also myLastAction...
				if (myLastAction == null) {
					// dumpDistancesToLog(fRound++);
					lAction = proposeInitialBid();
				} else
					// simply repeat last action
					lAction = myLastAction;
				break;
			}
		} catch (Exception e) {
			log("Exception in chooseAction:" + e.getMessage());
			e.printStackTrace();
			lAction = new Offer(getAgentID(), myLastBid);
		}
		myLastAction = lAction;
		{
			myPreviousBids.add(((Offer) myLastAction).getBid());
			myLastBid = ((Offer) myLastAction).getBid();
		}
		return lAction;
	}

	protected ACTIONTYPE getActionType(Action lAction) {
		ACTIONTYPE lActionType = ACTIONTYPE.START;
		if (lAction instanceof Offer)
			lActionType = ACTIONTYPE.OFFER;
		else if (lAction instanceof Accept)
			lActionType = ACTIONTYPE.ACCEPT;
		else if (lAction instanceof EndNegotiation)
			lActionType = ACTIONTYPE.BREAKOFF;
		return lActionType;
	}

	private double getTargetUtility(double myUtility, double oppntUtility) {
		return myUtility - getConcessionFactor();
	}

	private double getConcessionFactor() {
		// The more the agent is willing to concess on its aspiration value, the
		// higher this factor.
		return CONCESSIONFACTOR;
	}

	/**
	 * Prints out debug information only if the fDebug = true
	 * 
	 * @param pMessage
	 *            - debug informaton to print
	 */
	private void log(String pMessage) {
		if (fDebug)
			System.out.println(pMessage);
	}

	/**
	 * This function determines the accept probability for an offer. At t=0 it
	 * will prefer high-utility offers. As t gets closer to 1, it will accept
	 * lower utility offers with increasing probability. it will never accept
	 * offers with utility 0.
	 * 
	 * @param u
	 *            is the utility
	 * @param t
	 *            is the time as fraction of the total available time (t=0 at
	 *            start, and t=1 at end time)
	 * @return the probability of an accept at time t
	 * @throws Exception
	 *             if you use wrong values for u or t.
	 * 
	 */
	protected double Paccept(double u, double t1) throws Exception {
		double t = t1 * t1 * t1; // get more relaxed more to the end.
		if (u < 0 || u > 1.05)
			throw new Exception("utility " + u + " outside [0,1]");
		if (t < 0 || t > 1)
			throw new Exception("time " + t + " outside [0,1]");
		if (u > 1.)
			u = 1.;

		if (t == 0.5)
			return u;
		return (u - 2. * u * t + 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u
				* (-1. + 2 * t))))
				/ (-1. + 2 * t);
	}

	private double sq(double x) {
		return x * x;
	}

	/*
	 * private double calculateEuclideanDistanceUtilitySpace(double[]
	 * pLearnedUtil, double[] pOpponentUtil) { double lDistance = 0; try {
	 * for(int i=0;i<pLearnedUtil.length;i++) lDistance = lDistance + sq(
	 * pOpponentUtil[i]-pLearnedUtil[i]); } catch (Exception e) {
	 * e.printStackTrace(); } lDistance = lDistance /
	 * utilitySpace.getDomain().getNumberOfPossibleBids(); return lDistance; }
	 * private double calculateEuclideanDistanceWeghts(double[] pExpectedWeight)
	 * { double lDistance = 0; int i=0; try { for(Issue lIssue :
	 * utilitySpace.getDomain().getIssues()) { lDistance = lDistance
	 * +sq(fNegotiation.getOpponentWeight(this, lIssue.getNumber())
	 * -pExpectedWeight[i]); i++; } } catch (Exception e) { e.printStackTrace();
	 * } return lDistance/(double)i; }
	 * 
	 * private double calculatePearsonDistanceUtilitySpace(double[]
	 * pLearnedUtility, double[] pOpponentUtil) { double lDistance = 0; double
	 * lAverageLearnedUtil=0; double lAverageOriginalUtil=0; //calculate average
	 * values for(int i=0;i<pLearnedUtility.length;i++) { lAverageLearnedUtil =
	 * lAverageLearnedUtil + pLearnedUtility[i]; lAverageOriginalUtil =
	 * lAverageOriginalUtil + pOpponentUtil[i]; } lAverageLearnedUtil =
	 * lAverageLearnedUtil
	 * /(double)(utilitySpace.getDomain().getNumberOfPossibleBids());
	 * lAverageOriginalUtil = lAverageOriginalUtil/
	 * (double)(utilitySpace.getDomain().getNumberOfPossibleBids()); //calculate
	 * the distance itself double lSumX=0; double lSumY=0; for(int
	 * i=0;i<pLearnedUtility.length;i++) { lDistance = lDistance +
	 * (pLearnedUtility[i]-lAverageLearnedUtil)*
	 * (pOpponentUtil[i]-lAverageOriginalUtil); lSumX = lSumX +
	 * sq(pLearnedUtility[i]-lAverageLearnedUtil); lSumY = lSumY +
	 * sq(pOpponentUtil[i]-lAverageOriginalUtil);
	 * 
	 * }
	 * 
	 * 
	 * return lDistance/(Math.sqrt(lSumX*lSumY)); }
	 * 
	 * private double calculatePearsonDistanceWeghts(double[] pExpectedWeight) {
	 * double lDistance = 0; double lAverageLearnedWeight=0; double
	 * lAverageOriginalWeight=0; int i=0; try { for(Issue lIssue :
	 * utilitySpace.getDomain().getIssues()) { lAverageLearnedWeight =
	 * lAverageLearnedWeight +pExpectedWeight[i]; lAverageOriginalWeight =
	 * lAverageOriginalWeight + fNegotiation.getOpponentWeight(this,
	 * lIssue.getNumber()); i++; } } catch (Exception e) { e.printStackTrace();
	 * } lAverageLearnedWeight = lAverageLearnedWeight/(double)(i);
	 * lAverageOriginalWeight= lAverageOriginalWeight/ (double)(i);
	 * 
	 * //calculate the distance itself i=0; double lSumX=0; double lSumY=0; try
	 * { for(Issue lIssue : utilitySpace.getDomain().getIssues()) { lDistance =
	 * lDistance +(fNegotiation.getOpponentWeight(this, lIssue.getNumber())-
	 * lAverageOriginalWeight)*(pExpectedWeight[i]-lAverageLearnedWeight); lSumX
	 * = lSumX + sq(fNegotiation.getOpponentWeight(this, lIssue.getNumber())-
	 * lAverageOriginalWeight); lSumY = lSumY +
	 * sq(pExpectedWeight[i]-lAverageLearnedWeight); i++; } } catch (Exception
	 * e) { e.printStackTrace(); }
	 * 
	 * return lDistance/(Math.sqrt(lSumX*lSumY)); } private double
	 * calculateRankingDistanceUtilitySpaceMonteCarlo(double[] pLearnedUtil,
	 * double[] pOpponentUtil) { double lDistance = 0; int lNumberOfPossibleBids
	 * = (int)(utilitySpace.getDomain().getNumberOfPossibleBids()); int
	 * lNumberOfComparisons = 10000000; for(int k=0;k<lNumberOfComparisons ;k++)
	 * { int i = (new Random()).nextInt(lNumberOfPossibleBids-1); int j = (new
	 * Random()).nextInt(lNumberOfPossibleBids-1);
	 * if(((pLearnedUtil[i]>pLearnedUtil
	 * [j])&&(pOpponentUtil[i]>pOpponentUtil[j]))||
	 * ((pLearnedUtil[i]<pLearnedUtil
	 * [j])&&(pOpponentUtil[i]<pOpponentUtil[j]))||
	 * ((pLearnedUtil[i]==pLearnedUtil
	 * [j])&&(pOpponentUtil[i]==pOpponentUtil[j]))) {
	 * 
	 * } else lDistance++;
	 * 
	 * } return ((double)lDistance)/((double)lNumberOfComparisons); } private
	 * double calculateRankingDistanceUtilitySpace(double[] pLearnedUtil,
	 * double[] pOpponentUtil) {
	 * 
	 * double lDistance = 0; int lNumberOfPossibleBids =
	 * (int)(utilitySpace.getDomain().getNumberOfPossibleBids());
	 * 
	 * try { for(int i=0;i<lNumberOfPossibleBids-1;i++) { for(int
	 * j=i+1;j<lNumberOfPossibleBids;j++) { //if(i==j) continue; if
	 * (Math.signum(
	 * pLearnedUtil[i]-pLearnedUtil[j])!=Math.signum(pOpponentUtil[i
	 * ]-pOpponentUtil[j])) lDistance++;
	 * 
	 * } //for } //for } catch (Exception e) { e.printStackTrace(); }
	 * 
	 * lDistance = 2 * lDistance /
	 * (utilitySpace.getDomain().getNumberOfPossibleBids
	 * ()*(utilitySpace.getDomain().getNumberOfPossibleBids())); return
	 * lDistance; } private double calculateRankingDistanceWeghts(double
	 * pExpectedWeights[]) { double lDistance = 0; double[] lOriginalWeights =
	 * new double[utilitySpace.getDomain().getIssues().size()]; int k=0; try {
	 * for(Issue lIssue : utilitySpace.getDomain().getIssues()) {
	 * lOriginalWeights[k] = fNegotiation.getOpponentWeight(this,
	 * lIssue.getNumber()); k++; } } catch (Exception e) { e.printStackTrace();
	 * } k=0; int nrOfIssues = utilitySpace.getDomain().getIssues().size();
	 * for(int i=0; i<nrOfIssues-1;i++) { for(int j=i+1;j<nrOfIssues;j++) { k++;
	 * double tmpWeightLearned = pExpectedWeights[i]; double tmpWeightOriginal =
	 * lOriginalWeights[i]; double tmpWeight2Learned = pExpectedWeights[j];
	 * double tmpWeight2Original = lOriginalWeights[j];
	 * if(((tmpWeightLearned>tmpWeight2Learned
	 * )&&(tmpWeightOriginal>tmpWeight2Original))||
	 * ((tmpWeightLearned<tmpWeight2Learned
	 * )&&(tmpWeightOriginal<tmpWeight2Original))||
	 * ((tmpWeightLearned==tmpWeight2Learned
	 * )&&(tmpWeightOriginal==tmpWeight2Original))) {
	 * 
	 * } else lDistance++;
	 * 
	 * } } return ((double)lDistance)/((double)k); }
	 * 
	 * protected void dumpDistancesToLog(int pRound) { if (fSkipDistanceCalc)
	 * return; System.out.print(getName() +
	 * ": calculating distance between the learned space and the original one ..."
	 * );
	 * 
	 * double lExpectedWeights[] = new
	 * double[utilitySpace.getDomain().getIssues().size()]; int i=0; for(Issue
	 * lIssue : utilitySpace.getDomain().getIssues()) {
	 * lExpectedWeights[i]=fOpponentModel.getExpectedWeight(i); i++; }
	 * 
	 * 
	 * double pLearnedUtil[] = new
	 * double[(int)(utilitySpace.getDomain().getNumberOfPossibleBids())]; //
	 * HashMap<Bid, Double> pLearnedSpace = new HashMap<Bid, Double>();
	 * BidIterator lIter = new BidIterator( utilitySpace.getDomain()); i=0;
	 * while(lIter.hasNext()) { Bid lBid = lIter.next(); try { pLearnedUtil[i]
	 * =fOpponentModel.getNormalizedUtility(lBid); // pLearnedSpace.put(lBid,
	 * new Double(pLearnedUtil[i]));
	 * 
	 * } catch (Exception e) { e.printStackTrace(); } i++; } double
	 * pOpponentUtil[] = new
	 * double[(int)(utilitySpace.getDomain().getNumberOfPossibleBids())]; //
	 * HashMap<Bid, Double> pOpponentSpace = new HashMap<Bid, Double>(); lIter =
	 * new BidIterator( utilitySpace.getDomain()); i=0; while(lIter.hasNext()) {
	 * Bid lBid = lIter.next(); try { pOpponentUtil[i] =
	 * fNegotiation.getOpponentUtility(this, lBid); // pOpponentSpace.put(lBid,
	 * new Double(pOpponentUtil[i])); } catch (Exception e) {
	 * e.printStackTrace(); } i++; }
	 * 
	 * double lEuclideanDistUtil =
	 * calculateEuclideanDistanceUtilitySpace(pLearnedUtil,pOpponentUtil);
	 * double lEuclideanDistWeights =
	 * calculateEuclideanDistanceWeghts(lExpectedWeights); double
	 * lRankingDistUtil = 0;
	 * if((int)(utilitySpace.getDomain().getNumberOfPossibleBids())>100000)
	 * lRankingDistUtil =
	 * calculateRankingDistanceUtilitySpaceMonteCarlo(pLearnedUtil,
	 * pOpponentUtil); else lRankingDistUtil =
	 * calculateRankingDistanceUtilitySpace(pLearnedUtil, pOpponentUtil); double
	 * lRankingDistWeights = calculateRankingDistanceWeghts(lExpectedWeights);
	 * double lPearsonDistUtil =
	 * calculatePearsonDistanceUtilitySpace(pLearnedUtil,pOpponentUtil); double
	 * lPearsonDistWeights = calculatePearsonDistanceWeghts(lExpectedWeights);
	 * SimpleElement lLearningPerformance = new
	 * SimpleElement("learning_performance");
	 * lLearningPerformance.setAttribute("round", String.valueOf(pRound));
	 * lLearningPerformance.setAttribute("euclidean_distance_utility_space",
	 * String.valueOf(lEuclideanDistUtil));
	 * lLearningPerformance.setAttribute("euclidean_distance_weights",
	 * String.valueOf(lEuclideanDistWeights));
	 * lLearningPerformance.setAttribute("ranking_distance_utility_space",
	 * String.valueOf(lRankingDistUtil));
	 * lLearningPerformance.setAttribute("ranking_distance_weights",
	 * String.valueOf(lRankingDistWeights));
	 * lLearningPerformance.setAttribute("pearson_distance_utility_space",
	 * String.valueOf(lPearsonDistUtil));
	 * lLearningPerformance.setAttribute("pearson_distance_weights",
	 * String.valueOf(lPearsonDistWeights)); System.out.println("Done!");
	 * System.out.println(lLearningPerformance.toString());
	 * fNegotiation.addAdditionalLog(lLearningPerformance);
	 * 
	 * }
	 */
	public final Bid getMaxUtilityBid() throws Exception {
		Bid maxBid = null;
		double maxutil = 0.;
		BidIterator bidit = new BidIterator(utilitySpace.getDomain());

		if (bidit.hasNext())
			maxBid = bidit.next();
		else
			throw new Exception("The domain does not contain any bids!");
		while (bidit.hasNext()) {
			Bid thisBid = bidit.next();
			if (fMarketPreassure)
				if (fNegotiation.getOpponentUtility(this, thisBid) < 0.3)
					continue;
			double thisutil = utilitySpace.getUtility(thisBid);
			if (thisutil > maxutil) {
				maxutil = thisutil;
				maxBid = thisBid;
			}
		}
		return maxBid;
	}

}
