package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2011.AgentK2SAS;

/**
 * This is the decoupled Offering Strategy for Agent K2 (ANAC2010). The code was
 * taken from the ANAC2011 AgentK2 and adapted to work within the BOA framework.
 * 
 * OPPONENT MODEL EXTENSION The strategy was extended to incorporate an opponent
 * model. Previously, in one case a random bid was chosen from a list of
 * candidates; this was replaced by selecting the best bid for the opponent. In
 * the other case, the strategy chooses the best bid for the opponent from the
 * set of all possible bids with a minimum target utility.
 * 
 * DEFAULT OM: None
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Mark Hendrikx
 */
public class AgentK2_Offering extends OfferingStrategy {

	private Random random200;
	private Random random300;
	private final boolean TEST_EQUIVALENCE = false;
	private SortedOutcomeSpace outcomespace;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AgentK2_Offering() {
	}

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel model, OMStrategy omStrategy,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		super.init(domainKnow, model, omStrategy, parameters);
		helper = new AgentK2SAS(negotiationSession);

		if (TEST_EQUIVALENCE) {
			random200 = new Random(200);
			random300 = new Random(300);
		} else {
			random200 = new Random();
			random300 = new Random();
		}

		if (!(opponentModel instanceof NoModel)) {
			outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		}
	}

	public BidDetails determineNextBid() {
		if (negotiationSession.getOpponentBidHistory().getHistory().size() > 0) {
			((AgentK2SAS) helper).calculateAcceptProbability();
		}

		ArrayList<BidDetails> bidTemp = new ArrayList<BidDetails>();

		for (Bid bid : ((AgentK2SAS) helper).getOfferedBidMap().keySet()) {
			double bidUtil = ((AgentK2SAS) helper).getOfferedBidMap().get(bid);
			if (bidUtil > ((AgentK2SAS) helper).getTarget()) {
				bidTemp.add(new BidDetails(bid, bidUtil, negotiationSession.getTime()));
			}
		}

		int size = bidTemp.size();

		if (size > 0) {
			if (opponentModel instanceof NoModel) {
				int sindex = (int) Math.floor(random200.nextDouble() * size);
				nextBid = bidTemp.get(sindex);
			} else {
				nextBid = omStrategy.getBid(bidTemp);
			}
		} else {
			double searchUtil = 0.0;
			if (opponentModel instanceof NoModel) {
				try {
					int loop = 0;
					while (searchUtil < ((AgentK2SAS) helper).getBidTarget()) {
						if (loop > 500) {
							((AgentK2SAS) helper).decrementBidTarget(0.01);
							loop = 0;
						}
						nextBid = searchBid();
						searchUtil = nextBid.getMyUndiscountedUtil();
						loop++;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			} else {
				nextBid = omStrategy.getBid(outcomespace, ((AgentK2SAS) helper).getBidTarget());
			}
		}
		return nextBid;
	}

	private BidDetails searchBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = negotiationSession.getUtilitySpace().getDomain().getIssues();

		BidDetails bid = null;

		for (Issue lIssue : issues) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				int optionIndex = random300.nextInt(lIssueDiscrete.getNumberOfValues());
				values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				int optionInd = random300.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
				values.put(lIssueReal.getNumber(),
						new ValueReal(lIssueReal.getLowerBound()
								+ (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) * (double) (optionInd)
										/ (double) (lIssueReal.getNumberOfDiscretizationSteps())));
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				int optionIndex2 = lIssueInteger.getLowerBound()
						+ random300.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
				values.put(lIssueInteger.getNumber(), new ValueInteger(optionIndex2));
				break;
			default:
				throw new Exception("issue type " + lIssue.getType() + " not supported by SimpleAgent2");
			}
		}
		Bid newBid = new Bid(negotiationSession.getUtilitySpace().getDomain(), values);
		bid = new BidDetails(newBid, negotiationSession.getUtilitySpace().getUtility(newBid),
				negotiationSession.getTime());
		return bid;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public String getName() {
		return "2011 - AgentK2";
	}
}