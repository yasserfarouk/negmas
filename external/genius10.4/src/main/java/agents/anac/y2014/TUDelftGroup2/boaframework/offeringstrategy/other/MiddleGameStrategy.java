package agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.misc.Pair;

/**
 * Just repeats the top 25 best bids. Best bids are defined as bids by utilrange
 * (currently 0.9 to 1.0). The point of this is to give our agent some time to
 * model the opponent
 */
public class MiddleGameStrategy extends AbstractBiddingStrategy {

	/** index of the bid to consider now */
	int currentBidIndex = 0;
	/**
	 * Top {@code maxNumberOfBids} in range (or less if there weren't that many)
	 */
	List<BidDetails> filtered_possibleBids = null;

	private BidSpaceExtractor_middle bidSpaceExtractor;
	private int starttime;
	private Bid lastbid;
	// private double e = 0.02; // Boulware
	private double e = 0.10; // lesser Boulware
	private int k = 0;
	private Bid stored_bid;
	private Bid next_bid;
	private List<Issue> domain_issues;
	private HashMap<Integer, Pair<Integer, Integer>> issueRanges;
	private int min_issues;
	private int max_issues;
	private int internal_bidcounter;

	MiddleGameStrategy(NegotiationSession negotiationSession,
			OpponentModel opponentModel) {
		super(negotiationSession, opponentModel);

		bidSpaceExtractor = new BidSpaceExtractor_middle();
		try {
			bidSpaceExtractor.init(negotiationSession, opponentModel, null,
					null);

		} catch (Exception e) {

			e.printStackTrace();
		}

		starttime = this.negotiationSession.getOwnBidHistory().getHistory()
				.size();
		lastbid = null;
		stored_bid = null;
		next_bid = null;
		internal_bidcounter = 0;

		domain_issues = negotiationSession.getDomain().getIssues();

		// GET VALUE RANGE FOR EACH ISSUE
		// issueRanges is indexed by
		// issue_index
		//
		// and the values is a Pair containing Integers (min,max)
		issueRanges = new HashMap<Integer, Pair<Integer, Integer>>();

		min_issues = 999999999;
		max_issues = 0;

		for (Issue issue : domain_issues) {
			int an_index = issue.getNumber();

			// To know where the issue indexing starts
			if (an_index < min_issues)
				min_issues = an_index;
			if (an_index > max_issues)
				max_issues = an_index;

			// Get range, by casting to IssueInteger
			int lowbound = ((IssueInteger) issue).getLowerBound();
			int highbound = ((IssueInteger) issue).getUpperBound();
			// int highbound=
			// ((IssueInteger)issue).getHighestObjectiveNr(lowbound); // results
			// in WRONG CENTERING, but maybe worth keeping. (the origin + delta
			// is skewed by the issue index, instead of true (max-min) /2). (so:
			// (index-min)/2 )

			// Store range for this issue in the table 'issueRanges'
			issueRanges.put(an_index, new Pair<Integer, Integer>(lowbound,
					highbound));
		}
		// END of GET VALUE RANGE FOR EACH ISSUE

	}

	@Override
	// see parent
	Bid getBid() {

		// if possible bids for opening strategy have not yet been calculated,
		// do so
		if (filtered_possibleBids == null)
			filtered_possibleBids = calculatePossibleBids();

		Bid bid = null;

		double time = negotiationSession.getTime();
		double utilityGoal;
		utilityGoal = p(time);

		if (stored_bid == null) {
			stored_bid = deepcopyBid(getbid_withOM_filtering());
		}

		double actualUtil = 0.0;
		try {
			actualUtil = this.negotiationSession.getUtilitySpace().getUtility(
					stored_bid);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (actualUtil > utilityGoal) {
			// this bid is good
			// but can we propose a better one?

			if (next_bid == null) {
				// get yet another bid
				next_bid = deepcopyBid(getbid_withOM_filtering());
			}
			double nextUtil = 0.0;
			try {
				nextUtil = this.negotiationSession.getUtilitySpace()
						.getUtility(next_bid);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			if (nextUtil > utilityGoal) {
				bid = deepcopyBid(next_bid);
				stored_bid = deepcopyBid(bid);
				next_bid = deepcopyBid(getbid_withOM_filtering());
			} else {
				bid = deepcopyBid(stored_bid);
			}

		}

		// the next bid is too good, keep spamming current offer

		if (bid == null) {
			// failsafe
			return bidSpaceExtractor.put_expected_bid_into_bestpastexp()
					.getBid();
		}

		// filtered_possibleBids

		// Return bid
		return bid;
	}

	private Bid getbid_withOM_filtering() {
		Bid bid;
		bid = filter();

		// if ran out of good bids, retrigger start list generation
		if (bid == null) {
			bidSpaceExtractor = new BidSpaceExtractor_middle();

			try {
				bidSpaceExtractor.init(negotiationSession, opponentModel, null,
						null);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			filtered_possibleBids = calculatePossibleBids();

			System.out.println("Resampled");

			bid = filter();

			// If still didnt find anything
			if (bid == null) {

				System.out.println("OM filtered out evrything");
				bid = nofilter();
			}
		}
		return bid;
	}

	private Bid nofilter() {
		int counter = 0;
		boolean found_good_bid = false;
		Bid bid = null;

		// Now do a roundrobin conceder filtered by OM

		// Get bid at current index
		bid = filtered_possibleBids.get(currentBidIndex).getBid();
		// try {
		// double actualUtil = this.negotiationSession.
		// getUtilitySpace().getUtility(bid);
		// } catch (Exception e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		// double om_score = opponentModel.getBidEvaluation(bid);

		// Update index
		currentBidIndex = this.negotiationSession.getOwnBidHistory()
				.getHistory().size()
				+ counter % filtered_possibleBids.size();

		return bid;

	}

	private Bid filter() {
		internal_bidcounter++;
		int counter = 0;
		boolean found_good_bid = false;
		Bid bid = null;

		while (!found_good_bid) {

			// Now do a roundrobin conceder filtered by OM

			// Get bid at current index
			bid = filtered_possibleBids.get(currentBidIndex).getBid();
			double actualUtil;
			try {
				actualUtil = this.negotiationSession.getUtilitySpace()
						.getUtility(bid);
				double om_score = opponentModel.getBidEvaluation(bid);
				//
				// System.out.print(actualUtil);
				// System.out.print(",");
				// System.out.print(om_score);
				// System.out.println("");
				//
				double issue_mistake_accepted = 0.3; // 30 % error is ok
				int D = this.negotiationSession.getUtilitySpace().getDomain()
						.getIssues().size();
				double discrepancy = 1 / ((double) D);
				discrepancy = discrepancy / issue_mistake_accepted;
				if (actualUtil > (((om_score + discrepancy) * 1.1))) {
					found_good_bid = true;
				} else {
					// System.out.println("Didnt bid one");
				}

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// Update index
			counter++;
			currentBidIndex = (internal_bidcounter + counter)
					% filtered_possibleBids.size();
			// System.out.println(currentBidIndex);
			if (counter > filtered_possibleBids.size()) {
				break;
			}
		}

		return bid;

	}

	/**
	 * Calculate top {@code maxNumberOfBids} in range {@code utilityRange}
	 * 
	 * @return the possible bids
	 */
	List<BidDetails> calculatePossibleBids() {
		// generate all bids given by utilityRange
		// List<BidDetails> possibleBids =
		// this.negotiationSession.getOutcomeSpace().getBidsinRange(utilityRange);
		List<BidDetails> possibleBids = new ArrayList<BidDetails>();

		// HOW MANY BIDS TO PRECHARGE
		// THIS IS TAKEN FROM THE BIDEXTRACTOR, BUT WE HAVE TO RUN IT ONCE TO
		// GET IT.
		possibleBids.add(bidSpaceExtractor.determineNextBid());
		int extracted_bidspace_size = bidSpaceExtractor.listsize;

		// THIS SHOULD BE AROUND 10*ISSUECOUNT, so 300 for example domain 3, and
		// 100 for domain1

		for (int i = 0; i < extracted_bidspace_size - 1; i++) {
			possibleBids.add(bidSpaceExtractor.determineNextBid());
		}

		// Sort the possible Bids
		Collections.sort(possibleBids);

		return possibleBids;
	}

	/**
	 * Boulware reused from example code:
	 * 
	 * This is an abstract class used to implement a TimeDependentAgent Strategy
	 * adapted from [1] [1] S. Shaheen Fatima Michael Wooldridge Nicholas R.
	 * Jennings Optimal Negotiation Strategies for Agents with Incomplete
	 * Information http://eprints.ecs.soton.ac.uk/6151/1/atal01.pdf
	 * 
	 * The default strategy was extended to enable the usage of opponent models.
	 * 
	 * Note that this agent is not fully equivalent to the theoretical model,
	 * loading the domain may take some time, which may lead to the agent
	 * skipping the first bid. A better implementation is
	 * GeniusTimeDependent_Offering.
	 * 
	 * @author Alex Dirkzwager, Mark Hendrikx From [1]:
	 * 
	 *         A wide range of time dependent functions can be defined by
	 *         varying the way in which f(t) is computed. However, functions
	 *         must ensure that 0 <= f(t) <= 1, f(0) = k, and f(1) = 1.
	 * 
	 *         That is, the offer will always be between the value range, at the
	 *         beginning it will give the initial constant and when the deadline
	 *         is reached, it will offer the reservation value.
	 * 
	 *         For e = 0 (special case), it will behave as a Hardliner.
	 */
	public double f(double t) {
		if (e == 0)
			return k;
		double ft = k + (1 - k) * Math.pow(t, 1.0 / e);
		return ft;
	}

	/**
	 * Makes sure the target utility with in the acceptable range according to
	 * the domain Goes from Pmax to Pmin!
	 * 
	 * @param t
	 * @return double
	 */
	public double p(double t) {
		return (1 - f(t));
	}

	private BidDetails deepcopyBidDetails(BidDetails source) {

		try {
			return arrayAsBidDetails(bidDetailsAsArray(source));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private Bid deepcopyBid(Bid source) {

		try {
			return arrayAsBidDetails(bidAsArray(source)).getBid();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private Integer[] bidAsArray(Bid inputbid) {
		Integer[] bidAsArrayResult = new Integer[inputbid.getValues().size()];
		int bidc = 0;
		// for (Value val : inputbid.getValues().values()) {
		// bidAsArrayResult[bidc]=((ValueInteger) val ).getValue();
		// bidc++;
		// }

		// for (Integer val : inputbid.getValues().keySet()) {

		for (int val = this.min_issues; val <= this.max_issues; val++) {
			try {
				bidAsArrayResult[bidc] = ((ValueInteger) inputbid.getValue(val))
						.getValue();

				// System.out.println(val);
			} catch (Exception e) {
				e.printStackTrace();
			}
			bidc++;
		}

		return bidAsArrayResult;
	}

	private Integer[] bidDetailsAsArray(BidDetails inputbid) {
		return bidAsArray(inputbid.getBid());
	}

	private BidDetails arrayAsBidDetails(Integer[] ibid) throws Exception {
		HashMap<java.lang.Integer, Value> bidP = new HashMap<Integer, Value>();

		for (int j = 0; j <= (this.max_issues - this.min_issues); j++) // i is
																		// the
																		// index
																		// onto
																		// the
																		// Issues
		{
			bidP.put(j + this.min_issues, new ValueInteger(ibid[j]));
		}

		// convert format
		Bid bid = new Bid(negotiationSession.getDomain(), bidP);
		double actualUtil = negotiationSession.getUtilitySpace()
				.getUtility(bid);
		BidDetails bidDetails = new BidDetails(bid, actualUtil);
		return bidDetails;
	}

}
