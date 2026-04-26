package agents.anac.y2014.TUDelftGroup2.boaframework.opponentmodel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.misc.Pair;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * First prototype of our custom implementation of {@code OpponentModel}.
 * <p>
 * This {@code Group2_OM} uses a straightforward frequency analysis algorithm
 * for weights estimation. For calculating the evaluation function of the
 * different options ({@code ValueDiscrete}) a simple counter is used.
 * 
 * @since 2013-12-13
 * @version 1.0
 */
public class Group2_OM extends OpponentModel {
	/** The amount of issues in this domain. */
	private int amountOfIssues;

	/** The issues of this domain as {@code IssueDiscrete} objects */
	private List<IssueInteger> issues;

	// matlab link stuff
	private int i_matlablink_acc_index;
	private int pointerom;

	private HashMap<Integer, Pair<Integer, Integer>> issueRanges;

	private List<Issue> domain_issues;

	private int min_issues;

	private int max_issues;

	private HashMap<Integer, HashMap<Integer, Integer>> histogram;

	/**
	 * Default initialization function for the {@code OpponentModel} Framework
	 * <p>
	 * This Initialization function sets the issue weights to equal values that
	 * sum up to 1 and the option evaluations to 1, which means they are all
	 * equally important
	 * 
	 * @param negotiationSession
	 *            {Provided by framework} This contains all the session
	 *            variables
	 * @param parameters
	 *            {Provided by framework}
	 */
	@Override
	public void init(NegotiationSession negotiationSession, Map<String, Double> parameters) {
		// wrap in a try catch to catch any possible exception
		try {

			// Initialize all variables used in object scope
			this.negotiationSession = negotiationSession;

			// Wouter: #1158 this does not work anymore with nonlinear space.
			// Replaced it with a created Additive domain.
			// (AdditiveUtilitySpace) negotiationSession
			// .getUtilitySpace().copy();
			this.opponentUtilitySpace = new AdditiveUtilitySpace(negotiationSession.getDomain());

			this.amountOfIssues = opponentUtilitySpace.getDomain().getIssues().size();

			// NONLINEAR modif
			// this.issues = new ArrayList<IssueDiscrete>();
			// this.issues = new ArrayList<IssueInteger>();

			// NONLINEAR modif
			// cast all issues to integer issues
			// for (Issue issue : opponentUtilitySpace.getDomain().getIssues())
			// issues.add((IssueInteger)issue);
			//

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
				// ((IssueInteger)issue).getHighestObjectiveNr(lowbound); //
				// results in WRONG CENTERING, but maybe worth keeping. (the
				// origin + delta is skewed by the issue index, instead of true
				// (max-min) /2). (so: (index-min)/2 )

				// Store range for this issue in the table 'issueRanges'
				issueRanges.put(an_index, new Pair<Integer, Integer>(lowbound, highbound));
			}
			// END of GET VALUE RANGE FOR EACH ISSUE

			Pair<Integer, Integer> bigyo = issueRanges.get(max_issues - min_issues + 1);
			int est_size = bigyo.getSecond() * max_issues;
			histogram = new HashMap<Integer, HashMap<Integer, Integer>>(est_size);

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
				// ((IssueInteger)issue).getHighestObjectiveNr(lowbound); //
				// results in WRONG CENTERING, but maybe worth keeping. (the
				// origin + delta is skewed by the issue index, instead of true
				// (max-min) /2). (so: (index-min)/2 )

				histogram.put(an_index, new HashMap<Integer, Integer>(highbound - lowbound + 1));
				for (int o = lowbound; o <= highbound; o++) {
					histogram.get(an_index).put(o, new Integer(0));
				}

			}

		}
		// Catch any exception that might occur.
		catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * Updates the opponent model given a bid and time.
	 */
	@Override
	public void updateModel(Bid opponentBid, double time) {
		// NONLINEAR modif
		// linear_model_update();

		// Add bid to histogram

		Integer[] op = bidAsArray(opponentBid);

		add_to_histogram(op);

	}

	private void add_to_histogram(Integer[] op) {

		for (Issue issue : domain_issues) {
			int issue_index = issue.getNumber();

			// Get range, by casting to IssueInteger
			int lowbound = ((IssueInteger) issue).getLowerBound();
			int highbound = ((IssueInteger) issue).getUpperBound();
			// int highbound=
			// ((IssueInteger)issue).getHighestObjectiveNr(lowbound); // results
			// in WRONG CENTERING, but maybe worth keeping. (the origin + delta
			// is skewed by the issue index, instead of true (max-min) /2). (so:
			// (index-min)/2 )

			int option = op[issue_index - min_issues];
			int acc = histogram.get(issue_index).get(option);
			acc = acc + 1;
			histogram.get(issue_index).remove(option);
			histogram.get(issue_index).put(option, new Integer(acc));

		}

	}

	@Override
	public double getBidEvaluation(Bid bid) {

		// TODO Override bid evaluation.

		int counter = 0;

		Integer[] op = bidAsArray(bid);
		for (Issue issue : domain_issues) {
			int issue_index = issue.getNumber();

			// Get range, by casting to IssueInteger
			int lowbound = ((IssueInteger) issue).getLowerBound();
			int highbound = ((IssueInteger) issue).getUpperBound();
			// int highbound=
			// ((IssueInteger)issue).getHighestObjectiveNr(lowbound); // results
			// in WRONG CENTERING, but maybe worth keeping. (the origin + delta
			// is skewed by the issue index, instead of true (max-min) /2). (so:
			// (index-min)/2 )

			int option = op[issue_index - min_issues];
			int acc = histogram.get(issue_index).get(option);

			// If the option in this issue was used more then expected, then say
			// that this bid, in this issue, should be considered a good bid.
			int NoOptions = highbound - lowbound + 1;
			int limit = this.negotiationSession.getOpponentBidHistory().size() * 1 / (NoOptions + 1);

			if (acc > limit) {
				// this issue is a hit
				// increse hit counter

				counter++;

			}

		}

		double matching_bid_measure = ((double) counter) / ((double) (this.max_issues - this.min_issues + 1));

		return matching_bid_measure;

		// NONLINEAR modif
		// return linear_getbideval(bid);
	}

	/**
	 * The model's name
	 * 
	 * @return "Prototype Opponent Model";
	 */
	@Override
	public String getName() {
		return "Prototype Opponent Model";
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
				bidAsArrayResult[bidc] = ((ValueInteger) inputbid.getValue(val)).getValue();

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
		double actualUtil = negotiationSession.getUtilitySpace().getUtility(bid);
		BidDetails bidDetails = new BidDetails(bid, actualUtil);
		return bidDetails;
	}

}