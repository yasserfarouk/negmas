/*
 * SimpleAgent.java
 *
 * Created on November 6, 2006, 9:55 AM
 *
 */

package agents;

import java.util.HashMap;
import java.util.List;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorReal;

/**
 * 
 * @author Dmytro Tykhonov & Koen Hindriks
 */

public class ABMPAgent extends Agent {
	private Action messageOpponent;
	// private int nrOfIssues;
	private Bid myLastBid = null;
	private Action myLastAction = null;

	// private double[] fIssueWeight;
	private enum ACTIONTYPE {
		START, OFFER, ACCEPT, BREAKOFF
	};

	// Paraters used in ABMP strategy
	// TODO: Include these parameters as agent parameters in agent's utility
	// template.
	// QUESTION: How to do that nicely, since these parameters are strategy
	// specific?
	private static final double NEGOTIATIONSPEED = 0.1; // TODO: Probably still
														// somewhere a bug. When
														// you set this too low
														// (e.g. 0.05), no deal
														// is reached and no
														// concession is done!
	private static final double CONCESSIONFACTOR = 1;
	private static final double CONFTOLERANCE = 0;
	private static final double UTIlITYGAPSIZE = 0.02; // Accept when utility
														// gap is <=
														// UTILITYGAPSIZE.

	// CHECK: Utility gap size needed since concession steps get VERY small when
	// opponent's last bid utility is
	// close to own last bid utility.

	// Code is independent from AMPO vs CITY case, but some specifics about
	// using this case as test are specified below.
	// ****************************************************************************************************
	// AMPO VS CITY: Outcome space has size of about 7 milion.
	// ****************************************************************************************************
	// ********************************************************
	// *******************************************
	// CHECK: ABMP gets stuck on the Car Example with a negotiation speed of
	// less than 0.05!!
	// ABMP "gets stuck" on AMPO vs CITY. The search through the space is not
	// effective in discrete outcome
	// spaces. Even with very high negotiation speed parameters (near 1) no bid
	// can be found with the target utility
	// at a certain point. In a discrete space, the evaluation distance between
	// two different values on an
	// issue need to be taken into account, which may differ from value to
	// value... In such spaces one strategy
	// would be to consider which combination of concessions on a set of issues
	// would provide
	// ********************************************************
	// *******************************************

	/** Creates a new instance of MyAgent */

	public ABMPAgent() {
		super();
	}

	public void init() {
		messageOpponent = null;
		myLastBid = null;
		myLastAction = null;

	}

	public void ReceiveMessage(Action opponentAction) {
		messageOpponent = opponentAction;
	}

	private Action proposeInitialBid() {
		Bid lBid = null;

		// Return (one of the) possible bid(s) with maximal utility.
		try {
			lBid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			e.printStackTrace();
		}
		myLastBid = lBid;
		return new Offer(getAgentID(), lBid);
	}

	private Action proposeNextBid(Bid lOppntBid) throws Exception {
		Bid lBid = null;
		double lMyUtility, lOppntUtility, lTargetUtility;
		// Both parties have made an initial bid. Compute associated utilities
		// from my point of view.
		lMyUtility = utilitySpace.getUtility(myLastBid);
		lOppntUtility = utilitySpace.getUtility(lOppntBid);
		lTargetUtility = getTargetUtility(lMyUtility, lOppntUtility);
		lBid = getBidABMPsimple(lTargetUtility);
		myLastBid = lBid;
		return new Offer(getAgentID(), lBid);
	}

	public Action chooseAction() {
		Action lAction = null;
		ACTIONTYPE lActionType;
		Bid lOppntBid = null;

		lActionType = getActionType(messageOpponent);
		switch (lActionType) {
		case OFFER: // Offer received from opponent
			lOppntBid = ((Offer) messageOpponent).getBid();
			if (myLastAction == null)
				// Other agent started, lets propose my initial bid.
				lAction = proposeInitialBid();
			else {
				try {
					if (utilitySpace.getUtility(lOppntBid) >= (utilitySpace
							.getUtility(myLastBid)) - UTIlITYGAPSIZE)
						// Opponent bids equally, or outbids my previous bid, so
						// lets accept.
						lAction = new Accept(getAgentID(), lOppntBid);
					else
						// Propose counteroffer. Get next bid.
						try {
							lAction = proposeNextBid(lOppntBid);
						} catch (Exception e) {
							e.printStackTrace();
						}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			break;
		case ACCEPT: // Presumably, opponent accepted last bid, but let's
						// check...
		case BREAKOFF:
			// nothing left to do. Negotiation ended, which should be checked by
			// Negotiator...
			break;
		default:
			// I am starting, but not sure whether Negotiator checks this, so
			// lets check also myLastAction...
			if (myLastAction == null)
				lAction = proposeInitialBid();
			else
				// simply repeat last action
				lAction = myLastAction;
			break;
		}

		myLastAction = lAction;
		return lAction;
	}

	private ACTIONTYPE getActionType(Action lAction) {
		ACTIONTYPE lActionType = ACTIONTYPE.START;

		if (lAction instanceof Offer)
			lActionType = ACTIONTYPE.OFFER;
		else if (lAction instanceof Accept)
			lActionType = ACTIONTYPE.ACCEPT;
		else if (lAction instanceof EndNegotiation)
			lActionType = ACTIONTYPE.BREAKOFF;
		return lActionType;
	}

	// ABMP Specific Code

	private Bid getBidABMPsimple(double targetUtility) throws Exception {
		// Value[] lIssueIndex = new Value[nrOfIssues];
		HashMap<Integer, Value> lIssueIndex = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		double[] lIssueAlpha = new double[issues.size()];
		double[] lBE = new double[issues.size()];
		double[] lBTE = new double[issues.size()];
		double[] lTE = new double[issues.size()];
		double lUtility = 0, lNF = 0, lAlpha, lUtilityGap, lTotalConcession = 0;

		// ASSUMPTION: Method computes a second bid. Method proposeInitialBid is
		// used to compute first bid.
		lUtilityGap = targetUtility - utilitySpace.getUtility(myLastBid);
		for (int i = 0; i < issues.size(); i++) {
			lBE[i] = (Double) (((AdditiveUtilitySpace) utilitySpace)
					.getEvaluator(issues.get(i).getNumber()).getEvaluation(
					((AdditiveUtilitySpace) utilitySpace), myLastBid, issues
							.get(i).getNumber()));
		}

		// STEP 1: Retrieve issue value for last bid and compute concession on
		// each issue.
		int i = 0;
		for (Issue lIssue : issues) {
			lAlpha = (1 - ((AdditiveUtilitySpace) utilitySpace)
					.getWeight(lIssue.getNumber())) * lBE[i]; // CHECK:
			// (1
			// -
			// lBE[i]);
			// This
			// factor
			// is
			// not
			// right??
			lNF = lNF
					+ ((AdditiveUtilitySpace) utilitySpace).getWeight(lIssue
							.getNumber()) * lAlpha;
			lIssueAlpha[i] = lAlpha;
			i++;
		}

		// Compute basic target evaluations per issue
		for (i = 0; i < issues.size(); i++) {
			lBTE[i] = lBE[i] + (lIssueAlpha[i] / lNF) * lUtilityGap;
		}

		// STEP 2: Add configuration tolerance for opponent's bid
		for (i = 0; i < issues.size(); i++) {
			lUtility = (Double) (((AdditiveUtilitySpace) utilitySpace)
					.getEvaluator(issues.get(i).getNumber()).getEvaluation(
					((AdditiveUtilitySpace) utilitySpace),
					((Offer) messageOpponent).getBid(), issues.get(i)
							.getNumber()));
			lTE[i] = (1 - CONFTOLERANCE) * lBTE[i] + CONFTOLERANCE * lUtility;
		}

		// STEP 3: Find bid in outcome space with issue target utilities
		// corresponding with those computed above.
		// ASSUMPTION: There is always a UNIQUE issue value with utility closest
		// to the target evaluation.
		// First determine new values for discrete-valued issues.
		double lEvalValue;
		int lNrOfRealIssues = 0;
		for (i = 0; i < issues.size(); i++) {
			lUtility = 1; // ASSUMPTION: Max utility = 1.
			Objective lIssue = issues.get(i);
			if (lIssue.getType() == ISSUETYPE.DISCRETE) {
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					lEvalValue = ((EvaluatorDiscrete) ((AdditiveUtilitySpace) utilitySpace)
							.getEvaluator(lIssue.getNumber()))
							.getEvaluation(lIssueDiscrete.getValue(j));
					if (Math.abs(lTE[i] - lEvalValue) < lUtility) {
						// lIssueIndex[i] = lIssueDiscrete.getValue(j);
						lIssueIndex.put(new Integer(lIssue.getNumber()),
								lIssueDiscrete.getValue(j));
						lUtility = Math.abs(lTE[i] - lEvalValue);
					}// if
				}// for

				lTotalConcession += ((AdditiveUtilitySpace) utilitySpace)
						.getWeight(lIssue.getNumber())
						* (lBE[i] - ((EvaluatorDiscrete) ((AdditiveUtilitySpace) utilitySpace)
								.getEvaluator(lIssue.getNumber()))
								.getEvaluation((ValueDiscrete) (lIssueIndex
										.get(lIssue.getNumber()))));
			} else if (lIssue.getType() == ISSUETYPE.REAL)
				lNrOfRealIssues += 1;
		}

		// TODO: Still need to integrate integer-valued issues somewhere here.
		// Low priority.

		// STEP 4: RECOMPUTE size of remaining concession step
		// Reason: Issue value may not provide exact match with basic target
		// evaluation value.
		// NOTE: This recomputation also includes any concession due to
		// configuration tolerance parameter...
		// First compute difference between actual concession on issue and
		// target evaluation.
		// TODO: Think about how to (re)distribute remaining concession over
		// MULTIPLE real issues. In car example
		// not important. Low priority.
		double lRestUtitility = lUtilityGap + lTotalConcession;
		// Distribute remaining utility of real issues. Integers still to be
		// done. See above.
		for (i = 0; i < issues.size(); i++) {
			Objective lIssue = issues.get(i);
			if (lIssue.getType() == ISSUETYPE.REAL) {
				lTE[i] += lRestUtitility / lNrOfRealIssues;
				EvaluatorReal lRealEvaluator = (EvaluatorReal) (((AdditiveUtilitySpace) utilitySpace)
						.getEvaluator(lIssue.getNumber()));
				double r = lRealEvaluator.getValueByEvaluation(lTE[i]);
				// lIssueIndex[i] = new ValueReal(r);
				lIssueIndex.put(new Integer(lIssue.getNumber()), new ValueReal(
						r));
			}
		}

		return new Bid(utilitySpace.getDomain(), lIssueIndex);
	}

	private double getTargetUtility(double myUtility, double oppntUtility) {
		return myUtility + getConcessionStep(myUtility, oppntUtility);
	}

	private double getNegotiationSpeed() {
		return NEGOTIATIONSPEED;
	}

	private double getConcessionFactor() {
		// The more the agent is willing to concess on its aspiration value, the
		// higher this factor.
		return CONCESSIONFACTOR;
	}

	private double getConcessionStep(double myUtility, double oppntUtility) {
		double lConcessionStep = 0, lMinUtility = 0, lUtilityGap = 0;

		// Compute concession step
		lMinUtility = 1 - getConcessionFactor();
		lUtilityGap = (oppntUtility - myUtility);
		lConcessionStep = getNegotiationSpeed() * (1 - lMinUtility / myUtility)
				* lUtilityGap;
		System.out.println(lConcessionStep);
		return lConcessionStep;
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}
}
