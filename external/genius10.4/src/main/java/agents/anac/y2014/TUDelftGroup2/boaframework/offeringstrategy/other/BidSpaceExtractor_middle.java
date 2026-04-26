package agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import agents.anac.y2014.TUDelftGroup2.boaframework.opponentmodel.Group2_OM;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.misc.Pair;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is class will try to explore the bidspace, and return various bids.
 * 
 * The goal of it is to serve as a means to create a small representation of the
 * true bid space. To aim is to cover the whole utilityspace as much as
 * possible, with as few bids as possible.
 * 
 * Exploration can be done various ways: +By sampling from a random distribution
 * over the issues +By hill climbing +By simulated annealing +By enumeration +a
 * hybrid of the above +or something entirely different
 * 
 * The output of a {@code BidSpaceExtractor} is a sequence of complete bids.
 * {@code determineNextBid} is used to obtain the bids one by one.
 * 
 * If you need all of the bids from this small bidspace, then you can just call
 * {@code determineNextBid} repeatedly.
 * 
 * {@code OfferingStrategy} is used as "interface" class, for the sake of
 * simplicity. A new kind of BidSpaceExtractor can be created by deriving from
 * OfferingStrategy and overriding {@code init} and {@code determineNextBid}.
 * 
 * 
 */
public class BidSpaceExtractor_middle extends OfferingStrategy {
	/** {@link AdditiveUtilitySpace} */
	AbstractUtilitySpace utilitySpace;

	// Used for MatLab link
	private int switchover_toMiddle;
	private int switchover_toEnd;

	// all used to get the timing of the failsafe right (used in the AC)
	int counter = 1;
	List<Double> timer = new ArrayList<Double>(10000);
	int thisNumBids;
	int thatNumBids;
	public int timing_numbids;
	public double timing_currentTime;
	public double timing_diffTime;
	public double timing_mean;
	public double timing_sd;
	public double timing_10avg;
	private List<Issue> domain_issues;
	private HashMap<Integer, Pair<Integer, Integer>> issueRanges;
	private int min_issues;
	private int max_issues;
	private Random random;
	private int counter_neighborhood;
	private List<BidDetails> best_past_exploration;
	private List<BidDetails> best_past_exploration_saved;
	private List<BidDetails> ownbidhistory;
	private int expo_step_size;
	private int resample_epoch;
	private int particle_count;
	private int exploration_length_initial;
	private int exploration_length;
	private int epoch_to_start_omni;
	private double range_pct;
	private String distrib;
	private int[][] bidso90 = { { 7, 6, 9, 3, 6, 6, 4, 3, 6, 7 }, { 7, 6, 9, 4, 6, 7, 5, 3, 7, 7 },
			{ 7, 6, 9, 4, 8, 7, 4, 4, 8, 7 }, { 7, 6, 9, 4, 8, 8, 4, 4, 8, 7 }, { 7, 6, 9, 5, 7, 8, 4, 5, 8, 6 },
			{ 7, 6, 9, 5, 7, 8, 5, 6, 8, 6 }, { 7, 6, 9, 5, 8, 8, 5, 5, 8, 7 }, { 7, 6, 9, 5, 8, 9, 5, 6, 8, 7 },
			{ 7, 6, 9, 6, 7, 9, 4, 5, 8, 6 }, { 7, 6, 9, 6, 7, 9, 5, 5, 8, 6 }, { 7, 7, 9, 3, 6, 6, 3, 3, 6, 6 },
			{ 7, 7, 9, 3, 6, 6, 4, 4, 7, 6 }, { 7, 7, 9, 3, 6, 6, 5, 3, 7, 7 }, { 7, 7, 9, 3, 6, 6, 5, 5, 7, 7 },
			{ 7, 7, 9, 3, 6, 6, 6, 4, 7, 7 }, { 7, 7, 9, 3, 7, 7, 6, 3, 8, 7 }, { 7, 7, 9, 4, 6, 6, 4, 4, 7, 7 },
			{ 7, 7, 9, 4, 6, 7, 4, 3, 7, 6 }, { 7, 7, 9, 4, 6, 7, 5, 4, 8, 6 }, { 7, 7, 9, 4, 7, 8, 4, 5, 8, 7 },
			{ 7, 7, 9, 4, 7, 8, 5, 4, 8, 6 }, { 7, 7, 9, 4, 7, 8, 5, 5, 8, 6 }, { 7, 7, 9, 5, 6, 7, 4, 4, 7, 7 },
			{ 7, 7, 9, 5, 6, 8, 4, 3, 8, 6 }, { 7, 7, 9, 5, 6, 8, 5, 3, 7, 6 }, { 7, 7, 9, 5, 7, 7, 4, 3, 8, 6 },
			{ 7, 7, 9, 5, 7, 9, 3, 5, 8, 7 }, { 7, 7, 9, 5, 7, 9, 4, 5, 8, 7 }, { 7, 7, 9, 5, 8, 8, 5, 5, 8, 7 },
			{ 7, 7, 9, 5, 8, 9, 5, 6, 8, 7 }, { 7, 7, 9, 6, 8, 8, 4, 5, 8, 7 }, { 7, 7, 9, 6, 8, 9, 5, 5, 8, 7 },
			{ 7, 8, 9, 5, 6, 8, 4, 4, 8, 7 }, { 8, 6, 9, 4, 6, 6, 4, 3, 6, 7 }, { 8, 6, 9, 4, 7, 7, 4, 4, 8, 6 },
			{ 8, 6, 9, 4, 7, 7, 4, 5, 8, 6 }, { 8, 6, 9, 4, 8, 8, 4, 4, 8, 7 }, { 8, 6, 9, 5, 7, 7, 4, 5, 8, 7 },
			{ 8, 6, 9, 5, 7, 8, 4, 5, 8, 6 }, { 8, 6, 9, 5, 9, 7, 4, 5, 8, 7 }, { 8, 6, 9, 6, 7, 9, 5, 6, 8, 6 },
			{ 8, 7, 9, 3, 7, 6, 6, 4, 8, 7 }, { 8, 7, 9, 4, 6, 6, 6, 4, 7, 7 }, { 8, 7, 9, 4, 6, 7, 4, 3, 7, 7 },
			{ 8, 7, 9, 4, 6, 7, 6, 4, 8, 8 }, { 8, 7, 9, 4, 7, 9, 6, 5, 8, 7 }, { 8, 7, 9, 4, 8, 8, 3, 5, 8, 7 },
			{ 8, 7, 9, 5, 6, 7, 5, 3, 7, 6 }, { 8, 7, 9, 5, 7, 7, 5, 4, 8, 7 }, { 8, 7, 9, 5, 7, 8, 3, 4, 8, 7 },
			{ 8, 7, 9, 5, 7, 8, 3, 5, 8, 7 }, { 8, 7, 9, 5, 7, 8, 5, 3, 8, 7 }, { 8, 7, 9, 5, 7, 8, 5, 5, 8, 6 },
			{ 8, 7, 9, 5, 7, 9, 4, 4, 8, 7 }, { 8, 7, 9, 5, 8, 8, 4, 4, 8, 7 }, { 8, 7, 9, 5, 9, 7, 5, 4, 8, 7 },
			{ 8, 7, 9, 6, 8, 8, 5, 6, 8, 6 }, { 8, 7, 9, 6, 8, 9, 4, 4, 8, 7 }, { 8, 7, 9, 6, 8, 9, 4, 5, 8, 7 },
			{ 8, 7, 9, 6, 8, 9, 5, 5, 8, 6 }, { 8, 8, 9, 4, 6, 6, 6, 4, 8, 7 }, { 8, 8, 9, 4, 6, 7, 4, 4, 8, 7 },
			{ 8, 8, 9, 4, 6, 7, 5, 5, 8, 7 }, { 9, 6, 9, 4, 8, 8, 4, 5, 8, 7 }, { 9, 6, 9, 4, 9, 8, 3, 5, 8, 7 },
			{ 9, 7, 9, 5, 7, 9, 4, 5, 8, 7 }, { 9, 7, 9, 5, 9, 7, 3, 5, 8, 7 }, { 9, 7, 9, 5, 9, 7, 4, 6, 8, 7 },
			{ 9, 7, 9, 6, 7, 9, 4, 5, 8, 7 } };
	private int top_how_much;

	private boolean neighbor_omni;

	private boolean tryanother_hood;

	private boolean already_set_up_the_nullspace;

	private BidDetails lastbid;

	private boolean do_restart;

	private boolean switch_to_opponentapproach;

	public int listsize;

	/**
	 * Empty constructor, guess BOA framework needs this.
	 */
	public BidSpaceExtractor_middle() {
	}

	/**
	 * Default contructor, gets some imputs from enviroment
	 * 
	 * @param negotiationSession
	 *            This is the Domain
	 * @param opponentModel
	 *            This is the model that we've implemented (in case of prototype
	 *            it is the {code {@link Group2_OM}).
	 * @param opponentModelStrategy
	 *            We won't be using this for the prototype.
	 * @param concessionFactor
	 *            This determines how easily the agent will make a lower bid.
	 * @param k
	 * @param maxTargetUtil
	 *            Maximum utility of the bid that will be made.
	 * @param minTargetUtil
	 *            Minimum utility of the bid that will be made.
	 */
	public BidSpaceExtractor_middle(NegotiationSession negotiationSession, OpponentModel opponentModel,
			OMStrategy opponentModelStrategy) {
		// this constructor is never actually called, the next will throw an
		// exception in case something calls it.
		if (true)
			try {
				throw new Exception("Assert failed");
			} catch (Exception e) {

				e.printStackTrace();
			}

	}

	/**
	 * Method which initializes the agent by setting all parameters.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel,
			OMStrategy opponentModelStrategy, Map<String, Double> parameters) throws Exception {

		// setup
		this.negotiationSession = negotiationSession;
		this.utilitySpace = negotiationSession.getUtilitySpace();
		this.opponentModel = opponentModel;
		this.omStrategy = opponentModelStrategy;
		domain_issues = negotiationSession.getDomain().getIssues();
		counter_neighborhood = 0;

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
			issueRanges.put(an_index, new Pair<Integer, Integer>(lowbound, highbound));
		}
		// END of GET VALUE RANGE FOR EACH ISSUE

		random = new Random(); // get a new pseudo-random number generator,
								// seeded with current time

		// this is just a bookmark to see on the side... All the parameters are
		// here

		// For methods that use particles, this parameters sets the number of
		// particles to use
		particle_count = 4;
		// particle_count = 1; // p

		// For some methods, it sets the amount of exploration to do between
		// resampling the particles
		// It is called initial, because it is used as a reference.
		exploration_length_initial = 18; // ex
		// exploration_length = exploration_length_initial*10;
		exploration_length = exploration_length_initial * 10;

		// Just a placeholder with memory
		best_past_exploration = new ArrayList<BidDetails>();

		// How far to look in neighborhood search. 1 is immediate neighbor, 2 is
		// the same direction but looking beyond the neighbor, 3 looks even
		// further, etc.
		// This will be adaptive in the future. For now it is set here, and then
		// decreased at any time. For example, decrease to 1 after the first
		// epoch of exploration is done.
		expo_step_size = 3; // during the first epoch only, after that it gets
							// reduced to 1. (see resampling phase)

		// Counts how many times there has been resampling of the particles.
		resample_epoch = 0; // just an initialized counter

		// At which point to we want to enable omni-directional search (NB. at
		// start we only search the 1st "quadrant" )
		// epoch_to_start_omni=259; // ep
		epoch_to_start_omni = 11; // ep p4 ex18 eomni11_was_259_before

		// Whether neighborhood should be explored 1-st quadrant or
		// omni-direcitonal way.
		// neighbor_omni=true;
		neighbor_omni = true;

		// Try an example another center of neighborhood? (so not the expected
		// value)
		tryanother_hood = false;

		// What is the lower threshold on utility, when resampling. 0.5 = take
		// the top 50% 0.8 = take the top 20% etc.
		range_pct = 0.5; // at first: lower threshold for range, when resampling
							// from top region. Then this can be increased to
							// 80%

		// skipsystem=0;

		// largestep=6561;

		// When "nearing" Pareto frontier, we switch from top 50% region, over
		// to taking the top 40 of all bids (in terms of own utility)
		top_how_much = 40;

		// A little storage so that our Extractor/Generator has some memory
		ownbidhistory = new ArrayList<BidDetails>();

		already_set_up_the_nullspace = false;

		// Examples:
		// Simulated annealing : cannot be made with a simple parametrization of
		// the particle filter here.
		// instead, think about it like this: the process homes onto regions of
		// good fit.

		// This is where the distribution method is selected. Uncomment one to
		// use.
		// distrib="gaussian";
		// distrib="neighborhood";
		// distrib="stoch_neighborhood";
		// distrib="uni";
		// distrib="gaussian";
		// distrib="resampled_stoch_neighborhood";
		// distrib="resampled_stoch_neighborhood_uni_style";
		// distrib="resampled_stoch_neighborhood_omni_end";
		// distrib="resampled_stoch_neighborhood_omni_end_corrected_particle";
		// distrib="enumeration";
		// distrib="bounty_over90";
		// distrib="resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop";
		// distrib="true_uniform";
		// distrib="matlab_optimizer";
		// distrib="across_the_space";
		// distrib="walk_a_path_to_meet";
		// distrib="look_along_axes";
		// distrib="competition_opening";
		distrib = "competition_middle";

		switch_to_opponentapproach = false;

		if (distrib == "matlab_optimizer") {
			// this.pre_calculations(null, null);
			this.setup_callbacks();
		}

		// step_across=0;
		// step_across_issue=

	}

	/**
	 * Not really useful for a BidSpaceExtractor but whatever.
	 */
	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	/**
	 * This method returns the next Bid to explore. Own utility is already
	 * calculated, and included.
	 * 
	 * The goal of it is to serve as a means to create a small representation of
	 * the true bid space. To aim is to cover the whole utilityspace as much as
	 * possible, with as few bids as possible.
	 * 
	 * A round counter is kept in {@code counter_neighborhood} to keep track of
	 * progress. Most mechanisms are driven by this counter +slowing the step
	 * size to 1 after a certain progress +switching from only increasing
	 * directions to omni-directional search +iterating a small list such as the
	 * particle list ( the counter mod N is used, where N is the size of the
	 * small list) +whatever you can invent
	 * 
	 * 
	 */
	@Override
	public BidDetails determineNextBid() {

		try {

			HashMap<java.lang.Integer, Value> bidP = new HashMap<Integer, Value>();
			String n_bid = "";

			// ASSEMBLE A RANDOM BID

			// Pre-calculation phase, that is outside the issue iteration loop.
			n_bid = pre_calculations(bidP, n_bid);
			// END OF PRE-PROCESSING

			// ENTERING PER ISSUE ITERATION LOOP.
			/**
			 * This iterates through each component in the bid. The
			 * distributions must return a value in 'a'. An 'a' is collected for
			 * each issue, and after the loop, assembled into a complete bid,
			 * called 'bidP'
			 * 
			 * 
			 */

			obtain_each_component(bidP, n_bid);

			// END OF ISSUE ITERATION
			//
			// 'bidP' now contains the bid.

			counter_neighborhood++; // if this becomes larger than the number of
									// "basis" then it automagically wraps
									// around (because of CharAt only takes the
									// first n bits, where n is the number of
									// issues)

			// convert format
			Bid bid = new Bid(negotiationSession.getDomain(), bidP);
			double actualUtil = utilitySpace.getUtility(bid);
			BidDetails bidDetails = new BidDetails(bid, actualUtil);

			// END OF ASSEMBLE BID

			// Save bid into local history
			ownbidhistory.add(bidDetails);

			// return the bid found by the current distribution/strategy
			return bidDetails;

		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}

	private String pre_calculations(HashMap<java.lang.Integer, Value> bidP, String n_bid) throws Exception {

		// Set distrib in INIT method

		if (distrib == "competition_opening") {

			// THIS STARTS AN ALTERNATE VARIABLE SEARCH FROM A RANDOM BID
			// AND ONLY OUTPUTS THE FINAL RESTING BID.

			if (counter_neighborhood == 0) {
				do_restart = false;
				put_expected_bid_into_bestpastexp();
				lastbid = null;

			} else {
				put_uni_random_bid_into_best_past_exploration();
			}

			List<BidDetails> path = get_trace_of_alternate_variable_search();

			best_past_exploration.clear();
			best_past_exploration.add(path.get(path.size() - 1));
			do_restart = true;

			if (counter_neighborhood > 500) {

				distrib = "competition_middle";
				counter_neighborhood = 0;
			}

		}

		if (distrib == "competition_middle") {

			// DO an AVS on the TOP 10 of the historical

			if (counter_neighborhood == 0) {
				do_restart = false;
				// PREPARE LIST

				// Sample whole history
				List<BidDetails> block_of_history_in_focus = this.negotiationSession.getOpponentBidHistory()
						.getHistory();

				// Take the top N bids
				top_how_much = 300;
				// top_how_much=10;
				BidHistory bh = new BidHistory(block_of_history_in_focus);
				block_of_history_in_focus = bh.getNBestBids(top_how_much);

				List<BidDetails> refined_list = explore_block_of_history_AVS_flat(block_of_history_in_focus);

				// refined_list;
				best_past_exploration.clear();
				best_past_exploration.addAll(refined_list);

				listsize = best_past_exploration.size();
				System.out.println(listsize);

				// THis list should now we filtered by the OM
				// best_past_exploration

			}

		}

		if (distrib == "nullspace around best bid") {

			// here we need to subsample even the 1.0 isoquant space, but even
			// that can be very large
			//
			//

			/**
			 * we could explore the two borders of the nullspace in an
			 * Issue.class
			 * 
			 * For each nullspace option ( say --\__ , where we have 2 options
			 * that keep us in nullspace) we take the borders (1,2, or more
			 * options can be in nullspace for an issue) Then we take all
			 * permutations. Expectedy, in 30 dimensions there will be 15*2 such
			 * borders. Which is actually 2^15=32k
			 * permuations....BidSpaceExtractor This is too much.
			 * 
			 * BETTER Start from one of our best bids. Look along each axis and
			 * make util map of issue take the issue that has the most options
			 * in nullspace Adapt to opponents first bid, concerning that issue,
			 * or as close as possible. In ONE move. No need to make multiple
			 * steps if we can stay in nullspace (unless one wants a very
			 * detailed map) do this in nullspace for all issues.
			 * 
			 * Expected result: we move inside the nullspace to the border that
			 * is the closest to opponents best Bid From here, we do a conceder,
			 * where we choose the issue that has the least cost of convergence
			 * first. In other words: we look at the cost of converging for each
			 * issue. We should select the issue that less painful to concede
			 * on.
			 * 
			 * Optionally, on the path of convergence, we can drop combing
			 * methods to find us the bids towards our maximum. With this, we
			 * expect the convergence line to be dragged to the right side for a
			 * few steps, as the method is working to find the best utility.
			 * This is recommended, with the aim of getting points from the
			 * Pareto line.
			 */
			if (already_set_up_the_nullspace == false) {

				already_set_up_the_nullspace = true;
				int NoSteps = 10;

				BidDetails first_opp_bid = this.negotiationSession.getOpponentBidHistory().getFirstBidDetails();

				Integer[] ownbid = bidAsArray(this.best_past_exploration.get(0).getBid());
				Integer[] opponentbid = bidAsArray(first_opp_bid.getBid());

				if (ownbid.length != opponentbid.length) {
					System.out.println("OWN AND OPPONENT HAVE DIFFERENT NUMBER OF ISSUES");
					return "";
				}

				Integer[] diff = getDiffOfTwoBids(ownbid, opponentbid);

				// PRINT DIFF as a string
				// printDiffOrBid(diff);

				// Obtain the vector that goes from one bid to another

				// 2 assumptions: that ordering matters, and that some 1.0 bids
				// lie on a surface in I+1 dimensions
				float[] tenth_of_diff = getInterpolationStepVector(NoSteps, diff);

				// Generate list

				// Move towards opponent in nullspace only

				// get bid from 'best_past_exploration'
				// Normally this will be a single bid, uploaded there by the
				// axes lookabout method.
				Bid prtcl = best_past_exploration.get(0).getBid();
				// Bid saved_prtcl= new Bid( negotiationSession.getDomain() ,
				// prtcl.getValues());

				// Map<Integer, List<Integer>> nullspace = get_nullspace(prtcl);

				// // This piles all of the visible nullspace onto
				// 'best_past_exploration'
				// for (int j = this.min_issues; j < this.max_issues; j++) {
				// List<Integer> null_options_in_this_issue = nullspace.get(j);
				// int saved_val = ((ValueInteger)
				// prtcl.getValue(j)).getValue();
				// for (Integer an_option : null_options_in_this_issue) {
				// //ValueInteger sd = (ValueInteger) prtcl.getValue(j);
				// prtcl.setValue(j,new ValueInteger(an_option));
				// best_past_exploration.add(new BidDetails( new Bid(prtcl),
				// utilitySpace.getUtility(prtcl)));
				// }
				// prtcl.setValue(j,new ValueInteger(saved_val));
				// }
				//
				// Make a kind of walk in the nullspace.

				// Iterate through issues,
				// choose a random null-option from that issue
				// set it,
				// post it onto exploration list 'best_past_exploration'
				// re-map surrounding axes
				// choose next issue and repeat
				// Loop multiple times through the whole set of issues, to hope
				// that enough *holes are traversed in the cheese*.
				// Remember that 270 points are taken for each set. (3*90)
				// Maybe 10 rounds ?

				// prtcl is the starting point
				// aVerySmallRandomStepInEachSingleIssueForAllIssues_looped_10_times(prtcl);

				// Now take very random steps, in multiple directions at once.
				// aManyStepsInRandomDirectionsAtOnce_looped_270_times(prtcl);
				// RSULTS: the same as the single direction.... this must be
				// because they cover the same space?

				// Next idea: start to converge towards a point
				// For simplicity just take the last bid of the opponent.

				// BidDetails bidDetails = arrayAsBid(opponentbid);

				// ANOTHER idea: just start the combing method from the
				// opponents best bid

				// DO A COMBING FROM OPPONENTS Best BID
				// prtcl = arrayAsBidDetails(opponentbid).getBid() ;
				// BidDetails prtcl_details =
				// this.negotiationSession.getOpponentBidHistory().getBestBidDetails();
				// BidDetails prtcl_details =
				// this.negotiationSession.getOpponentBidHistory().getFirstBidDetails();
				// BidDetails prtcl_details =
				// this.negotiationSession.getOpponentBidHistory().getLastBidDetails();

				// BidDetails prtcl_details =
				// this.negotiationSession.getOpponentBidHistory().getRandom();

				// // THIS IS USED IN MANY PLACES, UNCOMMENT IF IN DOUBT
				// // Pick a random particle
				//
				// BidDetails prtcl_details =
				// this.negotiationSession.getOpponentBidHistory().getHistory().get(random.nextInt(this.negotiationSession.getOpponentBidHistory().getHistory().size()
				// ));
				// prtcl = prtcl_details.getBid();
				//
				// best_past_exploration.clear();
				// best_past_exploration.add(
				// prtcl_details);

				// Start the Combing process, SLOW BID BY BID APPROACH
				// distrib="look_along_axes";

				// THIS IS THE SHOW COMBING ENDSPACE ONLY method

				distrib = "look_along_axes";

				List<BidDetails> path = new ArrayList<BidDetails>(this.max_issues);
				double valueoflast;
				boolean stoploop = false;

				while (stoploop == false) {
					List<BidDetails> asd = next_best_into_best_past_exploration();
					// asd.get(0)
					path.add(deepcopyBid(asd.get(0)));
					// new Bid( negotiationSession.getDomain() ,
					// asd.get(0).getBid().getValues()));

					if (path.size() > 1) {

						// COMPARE last 2 bids and if they are equal, then we
						// are stuck.
						if (Arrays.equals(

								bidAsArray(path.get(path.size() - 1).getBid())

								, bidAsArray(path.get(path.size() - 2).getBid())

						))
							stoploop = true;

					}
				}

				// System.out.println(path.size());
				// TRY MIDPOINTS IN EACH PATH
				// but take the side that is closer to us
				// int a = -4 + path.size()/2;

				best_past_exploration.clear();
				// best_past_exploration.add( path.get(a));
				best_past_exploration.addAll(path);

				// VERY NEXT AWESOME : JUST TAKE OUR ALRGE NEGHBORHOOD
				// INNULLSPACE, THEN TAKE A BUNCH OF NEIGHBORS THE USUAL WAY,
				// AND ONLY KEEP THESE NEW FINDINGS IN STORE. IF THE OPPONENT
				// PREF IS INHERENTLY CLOSE TO OUR POSITION, THEN WE CAN LIKE
				// THIS, GET A QUITE USABLE LIST, WITHOUT NEEDING ANY OPPONENT
				// MODEL

				// ALSO VERY AWESOME: HAVE AN OPPONENT AGNOSTIC PROPOSAL FOR
				// BIDS. THIS CAN BE THE BETTER KIND OF SAMPLING FROM NULLSPACE
				// OR SOMETHING ELSE. EG. DO A OLDSCHOOL STARTING LIST, THEN
				// TAKE A RANDOM SELECTION AND COMB IT OUT TO 1.0

				// APPROACH OPPONENTS BIDS method

				if (switch_to_opponentapproach) {

					// Sample whole history
					List<BidDetails> block_of_history_in_focus = this.negotiationSession.getOpponentBidHistory()
							.getHistory();

					// Sample first 1000 bids only
					int st = 0;
					int ed = 1000;
					// Clip at list size
					if ((block_of_history_in_focus.size() - 1) > ed) {
						ed = block_of_history_in_focus.size() - 1;
					}
					block_of_history_in_focus = block_of_history_in_focus.subList(st, ed);

					List<BidDetails> refined_list = explore_block_of_history_AVS_flat(block_of_history_in_focus);

					// refined_list;
					best_past_exploration.clear();
					best_past_exploration.addAll(refined_list);

					// System.out.println(path.size());
					// TRY MIDPOINTS IN EACH PATH
					// but take the side that is closer to us
					// int a = -4 + path.size()/2;
					//
					// best_past_exploration.clear();
					// best_past_exploration.add(
					// path.get(a));
					//

					// opponentbid.clone()

					// NEXT TRY: DOING THE QUICKER RANDOM WALK FOR SOME TIME,
					// AND THEN TAKING BEST (IN OUR VIEW) BID OF THE OPPONENT
					// AND DO THE COMBING PROCESS

					// BETTER: TAKE THE FIRST SOME THOUSAND BIDS OF THE
					// OPPONENT, AND TAKE 14 UNIFORM-RANDOM
					// SAMPLES. DO A COMBING APPROACH STARTING FROM EACH.
					// TAKE THOSE BIDS AS MIDDLE GAME LIST. THEN JUST PLAY
					// CONCEDER WITH IT.
					//
					// THE REASON BEHIND THIS: WE CANNOT BE SURE THAT THE
					// OPPONENT WILL PLAY ITS BEST BID
					// FIRST, SO WE ASSUME THAT IT IS EXPLORING, AND WE TAKE A
					// SAMPLE OF THE TRACE OF THAT
					// EXPLORATION.
					//
					// IN CASE THE OPP PLAYS ITS BEST BIDS, THEN THIS APPROACH
					// IS STILL VALID.

					// TRY MIDPOINTS IN EACH PATH

					// ! MUST TRY IN LINEAR SPACES AS WELL !

				}

			} else {

				// ALL TURNS AFTER SETUP TURN
				// do nothing

				// after having played the whole set,

				// // SWITCH to another method : look at best of opponents bid,
				// and then DO a combing process to my util 1.0.
				// if (((counter_neighborhood % best_past_exploration.size()) ==
				// 0) && (counter_neighborhood>0)){
				// prtcl = arrayAsBidDetails(opponentbid).getBid() ;
				//
				// best_past_exploration.clear();
				// best_past_exploration.add(
				// arrayAsBidDetails(
				// opponentbid));
				//
				// // Start the Combing process
				// distrib="look_along_axes";
				// }
				//

			}

		}

		if (distrib == "look_along_axes") {
			// we look along each axes, as far as we can see
			// this will take (No-1)*Ni, where No is the number of options, and
			// Ni is the number of issues
			// For domain 3 that will be 9*30=270
			// for domain 1 9*10=90;
			// so the method scales linearly with the number of dimensions.

			// algorithm

			// take the current bid (which E(I) at start)
			// take an issue
			// store util for each variation of current bid, where issue is
			// replaced by 0..9
			// Do this once for all issues

			// Find the best util in all these utility matrices (issue,option)

			// Update current bid by that option.
			// post this bid in 'best_past_exploration'
			// next, the per issue method will propose that bid

			// INITIALIZE with a single particle
			if (counter_neighborhood == 0) {
				do_restart = false;
				put_expected_bid_into_bestpastexp();
				lastbid = null;

			}

			// // THIS IS PART OF THE SHOW COMBING ENDZONE ONLY method
			//
			// if (do_restart){
			// // this means there is no more road forward
			// // restart at random point
			//
			// put_uni_random_bid_into_best_past_exploration();
			// lastbid=null;
			// }
			//
			// boolean stoploop = false;
			//
			// while (!stoploop){
			// next_best_into_best_past_exploration();
			// double best_util =
			// best_past_exploration.get(0).getMyUndiscountedUtil();
			//
			// if (lastbid!=null)
			// {
			//
			//
			// if (Arrays.equals(
			//
			// bidAsArray(
			// best_past_exploration.get(0).getBid()
			// )
			//
			//
			// ,bidAsArray(
			// lastbid.getBid()
			// )
			//
			//
			// )) {
			// stoploop=true;
			//
			//
			//
			//
			// }
			//
			//
			// }
			//
			// lastbid= deepcopyBid( best_past_exploration.get(0));
			//
			//
			//
			// // One of the 2 GREEDY algorithms
			// // if (best_util==1.0){
			// // //if (best_util>0.98){
			// //
			// // // Switch over to exploring the Nullspace around where we are.
			// // distrib="nullspace around best bid";
			// //
			// // already_set_up_the_nullspace=false;
			// //
			// // System.out.println(best_util);
			// //
			// //
			// // }
			//
			// // If it gets stuck in a local minimia, just rlaunch the particle
			// from a random place.
			//
			// // or just expplore the coplanar solutions by +1,1,,1,1,1,1
			// }
			//
			// // Keep output in 'best...explored', but flag a restart
			// do_restart=true;

			// THIS IS THE SIMPLE LOOK ALONG AXES x 2 method (aka
			// "alternate variable search")
			// It will generate a bid history with an AVS from E(I) and an AVS
			// from zero bid. (0,0,0,0,0,...)
			// This should help me see if the <<number of util significant steps
			// with AVS>> is any different of a metric than just lookng at my
			// own util of the opponent bid in question
			// RESULST: to be done. Expected results... the <<numb...>> will
			// correlate strongly with my util of the opponent bid.

			// So instead we just take the L0 norm on the diff of the opponent
			// bid and the mainstay of the opponents bids.
			// I get the mains....
			// Other: do disregard the issues that dont matter.

			// Other: if i take he NUSSAVS I can estime how close we are, on a
			// bid pair basis.

			// Treating issues kinda separately
			// I make a a binned counter for each issue
			//

			// We make a crude density estimator, by making a histogram for each
			// issue.
			// If an opponent bid has an issue with an "often" appeared option,
			// and all issues
			// are withing that range then then we are close to the nullspace of
			// the opponent.
			// The metric is how many issues are in nullspace from the bid. If
			// all, then give it util
			// 1 and if none we give it zero. In between we step by 1/NoIssues.

			// with an empty histoyry, what should this return? A: depends on
			// where the OM is used.
			// But I guess if we use an Openening strategy it will not happen.

			// IN DETAILS
			/**
			 * we make a histogram class in OM.update we add a new bid to
			 * Histogram
			 * 
			 * when readin OM: utility will be rough, but comparable to AVS step
			 * size
			 * 
			 * 
			 * 
			 * 
			 */

			//
			//
			//
			//
			//
			// if (do_restart){
			// // this means there is no more road forward
			// // restart at random point
			//
			// put_uni_random_bid_into_best_past_exploration();
			// put_zero_bid_into_best_past_exploration();
			// lastbid=null;
			// }
			//
			// boolean stoploop = false;
			//
			// while (!stoploop){
			// next_best_into_best_past_exploration();
			// double best_util =
			// best_past_exploration.get(0).getMyUndiscountedUtil();
			//
			// if (lastbid!=null)
			// {
			//
			//
			// if (Arrays.equals(
			//
			// bidAsArray(
			// best_past_exploration.get(0).getBid()
			// )
			//
			//
			// ,bidAsArray(
			// lastbid.getBid()
			// )
			//
			//
			// )) {
			// stoploop=true;
			//
			//
			//
			//
			// }
			//
			//
			// }
			//
			// lastbid= deepcopyBid( best_past_exploration.get(0));
			//
			//
			//
			// // One of the 2 GREEDY algorithms
			// // if (best_util==1.0){
			// // //if (best_util>0.98){
			// //
			// // // Switch over to exploring the Nullspace around where we are.
			// // distrib="nullspace around best bid";
			// //
			// // already_set_up_the_nullspace=false;
			// //
			// // System.out.println(best_util);
			// //
			// //
			// // }
			//
			// // If it gets stuck in a local minimia, just rlaunch the particle
			// from a random place.
			//
			// // or just expplore the coplanar solutions by +1,1,,1,1,1,1
			// }
			//
			// // Keep output in 'best...explored', but flag a restart
			// do_restart=true;

		}

		if (distrib == "across_the_space") {
			// step across 0,0,0,0, to 9,9,9,9,9, with steps in diagonal
			// 1,1,1,1,1,1,

			// at each step, look at all dimensions individually, and map a util
			// curve across it.

			// 9*9*9... = 90 steps for all issues
			// this is a local map
			// we record the local map for all steps betwwen 0,0,0, to 9,9,9,
			// which is 10 steps, so 900 steps all together.

			//

			int step = (counter_neighborhood % 10);

		}

		if (distrib == "matlab_optimizer") {
			// Thread.sleep(3000);

		}

		if (distrib == "neighborhood") {

			// We calculate the next neighbor to use :
			// We can use base 2 or base 3, depending if we want quadrant search
			// or omni-directional search

			// % It is all numbers between 0 and 2^n, where n is the number of
			// issues.
			// % in binary. Or for omni: all numbers between 0 and 3^n, where n
			// is the number of issues in base 3.

			// SETUP n_bid for quadrant search as a default
			// Pad left with 0
			// http://stackoverflow.com/questions/4469717/left-padding-a-string-with-zeros
			n_bid = agents.org.apache.commons.lang.StringUtils.leftPad(Integer.toBinaryString(counter_neighborhood),
					this.max_issues, "0");
			// We then split this, and this will let us know the neighborhood
			// around the zero vector.

			// SETUP n_bid for omni-directional search, if requested
			// True omnidirectional
			// Take a number between 0 and 3^10. Then express it in radix 3. By
			// subtracting 1, you get random numbers like -1 1 0 -1 1 1 -1 1 0
			// -1
			if (neighbor_omni)
				n_bid = agents.org.apache.commons.lang.StringUtils.leftPad(
						Integer.toString(random.nextInt((int) Math.floor(Math.pow(3, 10))), 3), this.max_issues, "0");
		}

		if (distrib == "true_uniform") {
			n_bid = agents.org.apache.commons.lang.StringUtils.leftPad(
					Integer.toString(random.nextInt((int) Math.floor(Math.pow(10, 10))), 10), this.max_issues, "0");

		}

		if (distrib == "enumeration") {

			// Try to get a shape for every issue, in every possible "context"
			// We sample the whole BidSpace to obtain a lower resolution
			// BidSpace
			// various sub-methods are possible:

			// NORMAL ENUMERATION
			// n_bid=org.apache.commons.lang.StringUtils.leftPad(
			// Integer.toString(counter_neighborhood) , this.max_issues, "0");

			// TURBO ENUMERATION
			// skip some steps, but it carries over to next issue
			// int trb_speed= 2; // try large prime numbers
			// n_bid=org.apache.commons.lang.StringUtils.leftPad(
			// Integer.toString(counter_neighborhood*trb_speed) ,
			// this.max_issues, "0");

			// SPARE ENUMERATION
			// This one also skips, but it doesnt carry over to next issue, so
			// for each issue we sample the same values.
			n_bid = agents.org.apache.commons.lang.StringUtils.leftPad(Integer.toString(counter_neighborhood, 3),
					this.max_issues, "0");
		}

		// if (distrib=="stoch_neighborhood"){
		// int sample_direction = random.nextInt((int) Math.round(Math.pow(2,
		// this.max_issues)));
		// n_bid=org.apache.commons.lang.StringUtils.leftPad(
		// Integer.toBinaryString(sample_direction) , this.max_issues, "0");
		// }

		// RESAMPLING OF THE STOCHASTIC AFTER 18 rounds
		if (distrib == "resampled_stoch_neighborhood" || distrib == "resampled_stoch_neighborhood_uni_style"
				|| distrib == "resampled_stoch_neighborhood_omni_end"
				|| distrib == "resampled_stoch_neighborhood_omni_end_corrected_particle"
				|| distrib == "resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop") {

			// For now, choose particles every 18 turns. Later this may be
			// adaptive and get large as we approach the Pareto (and we guess
			// this by our own utility (if it is high, then we are approaching
			// pareto) ; or we can guess it by the number of resamplings done)

			// int ee=exploration_length;

			// INITIALIZE with a single particle
			if (counter_neighborhood == 0) {
				best_past_exploration.clear();
				// construct the E(I) vector
				for (int i = this.min_issues; i <= this.max_issues; i++) {
					int a = -1;
					Pair<Integer, Integer> minmax = this.issueRanges.get(i);
					int mi = minmax.getFirst();
					int mx = minmax.getSecond();
					// Center on expected value of option
					a = (int) Math.round(a + (mi + mx) / 2);
					bidP.put(i, new ValueInteger(a));
				}

				// At this point bidP contains the center bid.
				// Now add to particle database
				Bid bid_start = new Bid(negotiationSession.getDomain(), bidP);
				// clear bidP
				bidP = new HashMap<Integer, Value>();
				double actualUtil = utilitySpace.getUtility(bid_start);
				BidDetails bidDetails_start = new BidDetails(bid_start, actualUtil);

				best_past_exploration.add(bidDetails_start);

			}

			// THIS IS EXPLORATION
			// default explores in 1st quadrant directions only
			int sample_direction = random.nextInt((int) Math.round(Math.pow(2, this.max_issues)));
			n_bid = agents.org.apache.commons.lang.StringUtils.leftPad(Integer.toBinaryString(sample_direction),
					this.max_issues, "0");

			// Enable true omni directional search (uniform sampling of
			// directions)
			if (neighbor_omni)
				n_bid = agents.org.apache.commons.lang.StringUtils.leftPad(
						Integer.toString(random.nextInt((int) Math.floor(Math.pow(3, 10))), 3), this.max_issues, "0");

			// THIS IS RESAMPLING aka choosing new particles.
			if ((counter_neighborhood % exploration_length) == 0) {
				if (counter_neighborhood > 0) { // skip the first turn, since we
												// already have a single
												// starting particle, and we
												// want to explore from there

					resample_epoch++;

					System.out.print("Resampling at ");
					System.out.println(counter_neighborhood);

					System.out.print("EPOCH= ");
					System.out.println(resample_epoch);

					// particle_count=particle_count*2;
					// NEIGHBOR EXPLORATION FOR EACH PARTICLE (expo length is
					// for all particles, so we have to increase it just to keep
					// the effective expo length constant)
					// this is also visible on the saved PNGs: 'Resampled
					// particles 1 sample around a particle take 1' when run on
					// 4000 round per side, this makes ~93 'epochs'
					// this is also visible on the saved PNGs: 'Resampled
					// particles 18 sample around a particle take 1' this is
					// what uses this: exploration_length =
					// exploration_length_initial*particle_count; when run on
					// 4000 round per side, this makes ~25 'epochs'
					// with this, the best result 0.91,0.83 : (Offer:
					// Bid[c1-i10: 6, c1-i9: 5, c1-i8: 9, c1-i7: 6, c1-i6: 5,
					// c1-i5: 8, c1-i4: 5, c1-i3: 4, c1-i2: 8, c1-i1: 6, ])
					// with this, the best result 0.82,0.90 : (Offer:
					// Bid[c1-i10: 4, c1-i9: 7, c1-i8: 6, c1-i7: 5, c1-i6: 6,
					// c1-i5: 9, c1-i4: 7, c1-i3: 6, c1-i2: 8, c1-i1: 7, ]) we
					// used parameters: particle_count = 1;
					// exploration_length_initial = 30; in a normal
					// 'resampled_stoch_neighborhood' (EPOCH was 54 rounds)
					// Adaptively increase the exploration time, so that each
					// particle has the same time available.
					exploration_length = exploration_length_initial * particle_count;

					// Also decrease the step size to 1
					// if (resample_epoch==1) expo_step_size=1;

					List<BidDetails> own_past_bids_recent_exploration;

					// HISTORY CAPABILITY when embedded in agent.
					List<BidDetails> own_past_bids_all = ownbidhistory;

					// How much of history to take into account?
					// +Most recent exploration
					// +all past exploration
					// Only take the fruits of the past exploration
					// if
					// (distrib=="resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop")
					// own_past_bids_recent_exploration =
					// own_past_bids_all.subList(own_past_bids_all.size()-exploration_length,
					// own_past_bids_all.size());
					// Just take all, and use the best from that.
					// if
					// (distrib!="resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop")
					// own_past_bids_recent_exploration = own_past_bids_all;
					own_past_bids_recent_exploration = own_past_bids_all;

					// if there are some "walls" or "mountains" in the utility
					// surface, then we can tunnel through it in some cases.
					// (because of the step size. If we step large, then by
					// accident we might find ourselves on the other side)

					// find best and worst in past exploration

					double min_u = 99999999;
					double max_u = -1;

					if (distrib != "resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop") {
						// THIS TAKES THE TOP 50% set by 'range_pct'
						for (BidDetails bidDetails : own_past_bids_recent_exploration) {
							if (bidDetails.getMyUndiscountedUtil() > max_u)
								max_u = bidDetails.getMyUndiscountedUtil();
							if (bidDetails.getMyUndiscountedUtil() < min_u)
								min_u = bidDetails.getMyUndiscountedUtil();
						}

						//
						double lower_threshold = min_u + (max_u - min_u) * range_pct;

						// Even more adaptively: narrow range as we approach 1.0
						// double lower_threshold=min_u+(max_u-min_u)*(max_u);

						// Now sample to get the particles

						// Collect the number of particles needed, but filter by
						// utility range
						// For this, we take a uniform sample.
						int n = own_past_bids_recent_exploration.size();

						int particles_selected = 0;
						// TODO empty the current particle list
						best_past_exploration.clear();

						// System.out.println("Lowbound===");
						// System.out.println(lower_threshold);

						while (particles_selected < particle_count) {
							// one from an Uniform distribution
							int idx_candidate_particle = random.nextInt(n);
							BidDetails candidate = own_past_bids_recent_exploration.get(idx_candidate_particle);
							if (candidate.getMyUndiscountedUtil() > lower_threshold) {
								// if we have a particle in the top 50%
								// Store particle parameters
								best_past_exploration.add(candidate);
								particles_selected++;
							}
						}
					} else // ...if
							// (distrib!="resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop")
					{
						// THIS TAKES THE TOP N bids set by 'top_how_much'

						BidHistory bh = new BidHistory(own_past_bids_recent_exploration);
						own_past_bids_recent_exploration = bh.getNBestBids(top_how_much);
						// Print the selected bids on console
						// System.out.println(own_past_bids_recent_exploration.size());
						// for (BidDetails bidDetails :
						// own_past_bids_recent_exploration) {
						// System.out.println(bidDetails.getBid().toString());
						// }

						// ENABLE TUNNELING
						// this will look further
						// if (resample_epoch>(this.epoch_to_start_omni+4))
						// expo_step_size=2;

						int n = own_past_bids_recent_exploration.size();
						int particles_selected = 0;
						best_past_exploration.clear();

						// NOW PICK particle_count number of bids from this TOP
						// N list

						while (particles_selected < particle_count) {
							// one from an Uniform distribution
							int idx_candidate_particle = random.nextInt(n);
							BidDetails candidate = own_past_bids_recent_exploration.get(idx_candidate_particle);

							best_past_exploration.add(candidate);
							particles_selected++;
						}

					}
				}
			}
		}
		return n_bid;
	}

	private List<BidDetails> explore_block_of_history_AVS_flat(List<BidDetails> block_of_history_in_focus)
			throws Exception {
		Bid prtcl;
		// From the currently selected range of history, do some uni random
		// samplings
		// For a domain with 30 issues we pick 15 such locations.
		// " " " with 10 issues we pick 5
		// this is a very rough heuristic, can probably be improved a lot.

		// we then trace the search path of these particles, until the algorithm
		// comes to a rest naturally.
		// We save these separately, or flattened into a single list.

		// Overwrites best_past_exploration
		// Can add flat results to best_... by doing
		// best_past_exploration.addAll(refined_list);

		// Number of issues / 2 is just a rough estimate, so that the method can
		// scale well with various domains.
		// int combing_particles_no = this.max_issues/2;
		int combing_particles_no = this.max_issues;

		List<BidDetails> refined_list = new ArrayList<BidDetails>(this.max_issues * combing_particles_no);

		for (int i = 0; i < combing_particles_no; i++) {

			// List<BidDetails> block_of_history_in_focus=
			// this.negotiationSession.getOpponentBidHistory().getHistory();

			BidDetails prtcl_details = block_of_history_in_focus.get(random.nextInt(block_of_history_in_focus.size()));
			prtcl = prtcl_details.getBid();

			best_past_exploration.clear();
			best_past_exploration.add(prtcl_details);

			// NOW JUST TRACE THE PATH BETWEEN THE OPPOSING BIDS, AND SAVE IT
			List<BidDetails> path = get_trace_of_alternate_variable_search();

			refined_list.addAll(path);
		}
		return refined_list;
	}

	private List<BidDetails> get_trace_of_alternate_variable_search() throws Exception {

		// The true input to this comes from the first element of
		// 'best_past_exploration'
		// From there, it does an 'alternate variable search', and saves every
		// step along the way.
		// It might not actually find the global optimum if the issues interact
		// too much.
		// But it will definitely Converge somewhere because it always takes the
		// issue/option
		// with the highest utility.
		// The stop condition is when it can no longer find an issue/option pair
		// with an even higher utility. This is detected when we have the same
		// Bid twice in a row.
		//
		// This method uses and modifies best_past_exploration.

		List<BidDetails> path = new ArrayList<BidDetails>(this.max_issues);

		boolean stoploop = false;

		while (stoploop == false) {
			List<BidDetails> asd = next_best_into_best_past_exploration();
			// asd.get(0)
			path.add(deepcopyBid(asd.get(0)));
			// new Bid( negotiationSession.getDomain() ,
			// asd.get(0).getBid().getValues()));

			if (path.size() > 1) {

				// COMPARE last 2 bids and if they are equal, then we are stuck.
				if (Arrays.equals(

						bidAsArray(path.get(path.size() - 1).getBid())

						, bidAsArray(path.get(path.size() - 2).getBid())

				))
					stoploop = true;

			}
		}
		return path;
	}

	private void put_zero_bid_into_best_past_exploration() throws Exception {
		HashMap<java.lang.Integer, Value> bidP;
		bidP = new HashMap<Integer, Value>();
		best_past_exploration.clear();

		// Zero bid ( 0,0,0,0,0,... )
		for (int i = this.min_issues; i <= this.max_issues; i++) {
			int a = 0;
			bidP.put(i, new ValueInteger(a));
		}

		// At this point bidP contains the random bid.
		// Now add to particle database
		Bid bid_start = new Bid(negotiationSession.getDomain(), bidP);
		double actualUtil = utilitySpace.getUtility(bid_start);
		BidDetails bidDetails_start = new BidDetails(bid_start, actualUtil);

		best_past_exploration.add(bidDetails_start);

	}

	private void put_uni_random_bid_into_best_past_exploration() throws Exception {

		HashMap<java.lang.Integer, Value> bidP;
		bidP = new HashMap<Integer, Value>();
		best_past_exploration.clear();

		// Uniform distrib
		for (int i = this.min_issues; i <= this.max_issues; i++) {
			int a = -1;
			Pair<Integer, Integer> minmax = this.issueRanges.get(i);
			int mi = minmax.getFirst();
			int mx = minmax.getSecond();

			a = (int) Math.round(random.nextInt(mx - mi) + mi);
			bidP.put(i, new ValueInteger(a));
		}

		// At this point bidP contains the random bid.
		// Now add to particle database
		Bid bid_start = new Bid(negotiationSession.getDomain(), bidP);
		// clear bidP
		bidP = new HashMap<Integer, Value>();
		double actualUtil = utilitySpace.getUtility(bid_start);
		BidDetails bidDetails_start = new BidDetails(bid_start, actualUtil);

		best_past_exploration.add(bidDetails_start);

	}

	public BidDetails put_expected_bid_into_bestpastexp() {
		// This method puts E(I) into our commonly used Bid variable called
		// 'best_past_exploration'
		// The expected value is used in the probability theory sense, so
		// for an issue with options 40..70 the expected value is 55.
		// This method is useful when we have absolutely no prior knowledge
		// about
		// the utility function, so we just start exploring around the expected
		// value.

		BidDetails bidDetails_start = null;
		try {
			HashMap<java.lang.Integer, Value> bidP;
			bidP = new HashMap<Integer, Value>();
			best_past_exploration.clear();
			bidP.clear();
			// construct the E(I) vector
			for (int i = this.min_issues; i <= this.max_issues; i++) {
				int a = -1;
				Pair<Integer, Integer> minmax = this.issueRanges.get(i);
				int mi = minmax.getFirst();
				int mx = minmax.getSecond();
				// Center on expected value of option
				a = (int) Math.round((mi + mx) / 2);
				bidP.put(i, new ValueInteger(a));
			}

			// At this point bidP contains the center bid.
			// Now add to particle database
			Bid bid_start = new Bid(negotiationSession.getDomain(), bidP);
			// clear bidP

			double actualUtil = utilitySpace.getUtility(bid_start);
			bidDetails_start = new BidDetails(bid_start, actualUtil);

			best_past_exploration.add(bidDetails_start);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return bidDetails_start;
	}

	private List<BidDetails> next_best_into_best_past_exploration() throws Exception {
		Bid prtcl = best_past_exploration.get(0).getBid();
		// Bid saved_prtcl= new Bid( negotiationSession.getDomain() ,
		// prtcl.getValues());

		Map<Integer, HashMap<Integer, Double>> map = axes_lookahead_maps(prtcl);

		// find best issue/option combo from what we have seen
		int best_idx = -1;
		double best_util = 0.0;
		int best_opt = -1;

		for (Integer i : map.keySet()) {

			for (Integer o : map.get(i).keySet()) {
				Double one_variation_util = map.get(i).get(o);
				if (one_variation_util > best_util) {
					best_idx = i;
					best_util = one_variation_util;
					best_opt = o;
				}
			}
		}
		// best_idx , best_opt now contains the best next step according to
		// combing.
		// we set it
		prtcl = prtcl.putValue(best_idx, new ValueInteger(best_opt));
		best_past_exploration.clear();
		best_past_exploration.add(new BidDetails(new Bid(prtcl), best_util));
		return best_past_exploration;
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
		double actualUtil = utilitySpace.getUtility(bid);
		BidDetails bidDetails = new BidDetails(bid, actualUtil);
		return bidDetails;
	}

	private void aVerySmallRandomStepInEachSingleIssueForAllIssues_looped_10_times(Bid prtcl) throws Exception {
		Map<Integer, List<Integer>> nullspace;
		for (int i = 0; i < 10; i++) {
			// This will do the whole set of issues once, and then loop.

			for (int j = this.min_issues; j < this.max_issues; j++) {

				// regresh map around where we are
				nullspace = get_nullspace(prtcl);

				List<Integer> null_options_in_this_issue = nullspace.get(j);
				if (null_options_in_this_issue.size() != 0) {

					Integer an_option = null_options_in_this_issue
							.get(random.nextInt(null_options_in_this_issue.size()));

					// ValueInteger sd = (ValueInteger) prtcl.getValue(j);
					prtcl = prtcl.putValue(j, new ValueInteger(an_option));
					best_past_exploration.add(new BidDetails(new Bid(prtcl), utilitySpace.getUtility(prtcl)));
				}

			}
		}
	}

	private void aManyStepsInRandomDirectionsAtOnce_looped_270_times(Bid prtcl) throws Exception {
		Map<Integer, List<Integer>> nullspace;
		for (int i = 0; i < 3; i++) {
			// This will do the whole set of issues once, and then loop.

			for (int j = this.min_issues; j < this.max_issues; j++) {

				// regresh map around where we are
				nullspace = get_nullspace(prtcl);

				List<Integer> null_options_in_this_issue = nullspace.get(j);
				if (null_options_in_this_issue.size() != 0) {

					Integer an_option = null_options_in_this_issue
							.get(random.nextInt(null_options_in_this_issue.size()));

					prtcl = prtcl.putValue(j, new ValueInteger(an_option));

				}
			}

			// store outside the inner loop
			best_past_exploration.add(new BidDetails(new Bid(prtcl), utilitySpace.getUtility(prtcl)));
		}
	}

	private void printDiffOrBid(Integer[] diff) {
		for (int i = 0; i < diff.length; i++) {
			System.out.print(diff[i].toString() + ", ");

		}
		System.out.println("");
	}

	private float[] getInterpolationStepVector(int NoSteps, Integer[] diff) {
		float[] tenth_of_diff = new float[diff.length];
		for (int i = 0; i < tenth_of_diff.length; i++) {
			tenth_of_diff[i] = ((float) diff[i]) / NoSteps;

		}
		return tenth_of_diff;
	}

	private Integer[] getDiffOfTwoBids(

			Integer[] ownbid, Integer[] opponentbid) {

		Integer[] diff = new Integer[ownbid.length];
		for (int i = 0; i < diff.length; i++) {
			diff[i] = ownbid[i] - opponentbid[i];
		}
		return diff;
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

	private Map<Integer, List<Integer>> get_nullspace(Bid prtcl) throws Exception {
		Map<Integer, HashMap<Integer, Double>> map = axes_lookahead_maps(prtcl);
		Map<Integer, List<Integer>> nullspace = new HashMap<Integer, List<Integer>>(this.max_issues);

		int NoIssues = this.max_issues - this.min_issues + 1; // number of
																// issues
		for (Integer i : map.keySet()) {
			List<Integer> inside_an_issue = new ArrayList<Integer>(NoIssues);
			nullspace.put(i, inside_an_issue);

			for (Integer o : map.get(i).keySet()) {
				Double one_variation_util = map.get(i).get(o);

				// are we in nullspace?
				if (one_variation_util == 1.0) {

					// save into list
					inside_an_issue.add(o); // hopefully this also adds it to
											// nullspace, b/c reference
											// addressing

				}
			}
		}
		return nullspace;
	}

	private Map<Integer, HashMap<Integer, Double>> axes_lookahead_maps(Bid prtcl) throws Exception {
		Map<Integer, HashMap<Integer, Double>> map = new HashMap<Integer, HashMap<Integer, Double>>();

		for (int i = this.min_issues; i <= this.max_issues; i++) {

			Pair<Integer, Integer> minmax = this.issueRanges.get(i);
			int mi = minmax.getFirst();
			int mx = minmax.getSecond();
			// Center on expected value of option
			HashMap<Integer, Double> single_issue_map = new HashMap<Integer, Double>();
			// int saved_option=((ValueInteger)prtcl.getValue(i)).getValue();

			int saved_option = ((ValueInteger) prtcl.getValue(i)).getValue();

			for (int j = mi; j <= mx; j++) {
				prtcl = prtcl.putValue(i, new ValueInteger(j));
				double util = utilitySpace.getUtility(prtcl);
				single_issue_map.put(j, util);

			}

			// save data
			map.put(i, single_issue_map);

			// restore this issue, to the original option, so that we explore
			// all directions from 1 center.
			prtcl = prtcl.putValue(i, new ValueInteger(saved_option));
		}
		return map;
	}

	private void obtain_each_component(HashMap<java.lang.Integer, Value> bidP, String n_bid) throws Exception {

		for (int i = this.min_issues; i <= this.max_issues; i++) // i is the
																	// index
																	// onto the
																	// Issues
		{
			Pair<Integer, Integer> minmax = this.issueRanges.get(i);
			int mi = minmax.getFirst();
			int mx = minmax.getSecond();

			// A random choice from all options in this issue
			// Choose distribution
			// Uniform
			// Gaussian
			// Multivariate Gaussian (ie. different Gaussian for each issue)
			// Distrub set a paragraph above

			int a = -1;
			// UNIFORM
			if (distrib == "uni") {
				// Get a sample from a uniform distribution within the valid
				// range for this issue.
				a = random.nextInt(mx + 1 - mi) + mi; // NextInt is inclusive of
														// 0 and exclusive of
														// the higherbound in
														// parameter, so we add
														// 1, and remove the
														// min.
			}

			if ((distrib == "competition_opening") || (distrib == "competition_middle")) {

				// We have the a good enough bid in 'best_past_exploration', we
				// bid that
				// Pull out the value for this issue from the Bid generated
				// pre-processing.

				BidDetails one_particle = best_past_exploration
						.get(counter_neighborhood % best_past_exploration.size());

				// System.out.println(new Integer(counter_neighborhood %
				// best_past_exploration.size()).toString() );
				// System.out.println(one_particle.getBid().toString());

				// Center on the particle. What is the coordinate for this
				// issue?
				a = ((ValueInteger) one_particle.getBid().getValue(i)).getValue();

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;

			}

			if (distrib == "nullspace around best bid") {
				// THIS STARTS ONLY WHEN CALLED BY LOOK_ALONG_AXES

				// We have the 1.0 bid in 'best_past_exploration', we keep
				// spamming it
				// Pull out the value for this issue from the Bid generated
				// pre-processing.

				BidDetails one_particle = best_past_exploration
						.get(counter_neighborhood % best_past_exploration.size());

				// System.out.println(new Integer(counter_neighborhood %
				// best_past_exploration.size()).toString() );
				// System.out.println(one_particle.getBid().toString());

				// Center on the particle. What is the coordinate for this
				// issue?
				a = ((ValueInteger) one_particle.getBid().getValue(i)).getValue();

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;

			}

			if (distrib == "look_along_axes") {

				// we look along each axes, as far as we can see
				// this will take (No-1)*Ni, where No is the number of options,
				// and Ni is the number of issues
				// For domain 3 that will be 9*30=270
				// for domain 1 9*10=90;
				// so the method scales linearly with the number of dimensions.

				// Pull out the value for this issue from the Bid generated
				// pre-processing.

				BidDetails one_particle = best_past_exploration
						.get(counter_neighborhood % best_past_exploration.size());

				// Center on the particle. What is the coordinate for this
				// issue?
				a = ((ValueInteger) one_particle.getBid().getValue(i)).getValue();

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;

			}

			if (distrib == "across_the_space") {

				// step across 0,0,0,0, to 9,9,9,9,9, with steps in diagonal
				// 1,1,1,1,1,1,

				// at each step, look at all dimensions individually, and map a
				// util curve across it.

				// 9*9*9... = 90 steps for all issues
				// this is a local map
				// we record the local map for all steps betwwen 0,0,0, to
				// 9,9,9,
				// which is 10 steps, so 900 steps all together.

				//
			}

			// BOUNTY over 90
			if (distrib == "bounty_over90") {
				// The bounty has some good bids, stored in a plain array of
				// option=bidso90[bid_index],[issue]
				// This is just one attempt at mapping the neighborhoods. NEXT:
				// try true omni directional sampling, for better results.
				// Just pull up a bid from the bounty, and use the relevant
				// component from it.
				// a=this.bidso90[counter_neighborhood%bidso90.length][i-1];
				// a=this.bidso90[3][i-1]; // <---- this is a very good bid,
				// near Kalai, but not on it.
				a = this.bidso90[21][i - 1]; // <---- this is a very good bid,
												// near Kalai, but not on it.

				// pick random direction
				// a = a+ (random.nextInt(3)-1); // -1 0 +1

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;

			}

			// UNIFORM with resampling, and particle filters
			if (distrib == "resampled_stoch_neighborhood_uni_style") {
				// Pull out the value for this issue from the Bid generated in
				// pre-processing.
				// This is 1 particle per cycle.
				BidDetails one_particle = best_past_exploration
						.get(counter_neighborhood % best_past_exploration.size());

				// GET THE RANDOM DIRECTION
				// System.out.println(n_bid);
				// a=Character.getNumericValue(
				// n_bid.charAt(i-this.min_issues));

				// HERE WE DONT USE DIRECTION, BUT A RANDOM DELTA AROUND THE
				// PARTICLE
				// RANGE: -3 .. +3 for now, but later this will be hooked up to
				// 'expo_step_size'
				int delta = random.nextInt(7) - 3; // 0 1 2 3 4 5 6 --> -3 -2 -1
													// 0 1 2 3

				// Add delta to coordinate of particle
				// NB. we deal with issues separately, so now it is only 1
				// delta, and one issue from the particle
				a = ((ValueInteger) one_particle.getBid().getValue(i)).getValue() + delta;

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;

			}

			// GAUSSIAN
			if (distrib == "gaussian") {
				// Gaussian centered on expected value of the issue.
				a = (int) Math.round(random.nextGaussian() + (mi + mx) / 2);
				// System.out.println((int) Math.round( (mi+mx)/2));

				// Gaussian centered arounf zero
				// a = (int) Math.round( random.nextGaussian());

				// Gaussian centered arounf 9,9,9,...
				// a = (int) Math.round( random.nextGaussian()+9);

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;
			}

			// NEIGHBORHOOD
			// Try all in the same L-infinite norm.
			//

			// For 3 issues: These share the same L-infinite distance from 0 0
			// 0, the distance is 1.
			//
			// 0 0 1
			// 0 1 0
			// 1 0 0
			// 1 1 0
			// 1 0 1
			// 0 1 1
			// 1 1 1

			// %% in other words
			// % It is all numbers between 0 and 2^n, where n is the number of
			// issues.
			// % in binary.

			if (distrib == "neighborhood" || distrib == "enumeration") {

				// Pull out the value for this issue from the Bid generated in
				// the pre-processing.
				// System.out.println(n_bid);
				a = Character.getNumericValue(n_bid.charAt(i - this.min_issues));

				// Center on zero
				// a=a+0;

				// True OMNI
				if (neighbor_omni)
					a = a - 1;

				// Multiply exploration distance
				if (distrib == "neighborhood")
					a = a * expo_step_size;

				// wrong?: Flip random sign to make it omni-directionaly
				// if (distrib=="neighborhood" ) a=a*(random.nextInt(2)*-1);

				// if (distrib=="enumeration"); // no need to modify for NORMAL
				// and TURBO
				if (distrib == "enumeration")
					a = a * 4; // FOR RADIX 3, we multuply by 4 to get 0 4 8 as
								// options for each issue

				// Center on example point or the expected value of issue
				if (distrib == "neighborhood") {
					if (tryanother_hood)
						a = a + bidso90[3][i - this.min_issues];
					else
						a = (int) Math.round(a + (mi + mx) / 2);
				}
				if (distrib == "enumeration")
					; // no need to modify

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;
			}

			// TRUE Uniform distribution. Make a enumeration of all bids, then
			// index it using a uniform distributed random number.
			if (distrib == "true_uniform") {
				// WHAT is it?

				// Aim: Make 10xuniform distrib less into a Gaussian.
				//
				// How? : Enumerate all possibilities in a list, and then pull a
				// uni random sample from that.
				//
				//
				// Why does this work?
				// ---------------------------
				//
				// When taking a uni- random option for each issue, the rest of
				// the issues tend to have
				// and expecte value centered around E(I). If we pick from the
				// list, we might truly have a uniform
				// distribution over the bidspace. (Something Centrality
				// theorem?)
				//
				// Implementation: take a random number between 1 and 10^10.
				// Then take each digit from its
				// decimal form. This gives a whole bid.
				//

				// RESULTS: this results in a different sampling, than the
				// per-issue uniform. This repeatedly looks like an oval, while
				// the other is a circle

				// Pull out the value for this issue from the Bid generated in
				// the pre-processing.
				a = Character.getNumericValue(n_bid.charAt(i - this.min_issues));

				// FOR NOW CLIPPING IS NOT REALLY NEEDED
				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;
			}

			// STOCHASTIC SAMPLING
			if (distrib == "stoch_neighborhood") {

				// Pull out the value for this issue from the Bid generated in
				// pre-processing.
				// System.out.println(n_bid);
				a = Character.getNumericValue(n_bid.charAt(i - this.min_issues));

				// Center on zero
				// a=a+0;

				// Multiply exploration distance
				a = a * 3;
				// Center on expected value of option
				a = (int) Math.round(a + (mi + mx) / 2);
				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;
			}

			// RESAMPLING OF THE STOCHASTIC AFTER 18 rounds
			if (distrib == "resampled_stoch_neighborhood") {
				// Pull out the value for this issue from the Bid generated
				// pre-processing.

				BidDetails one_particle = best_past_exploration
						.get(counter_neighborhood % best_past_exploration.size());

				// select component for this issue
				a = Character.getNumericValue(n_bid.charAt(i - this.min_issues));

				// Multiply exploration distance
				// So here this means, how far agent looks in the given
				// direction
				a = a * expo_step_size;

				// Center on the particle. What is the coordinate for this
				// issue?
				a = (int) Math.round(a + ((ValueInteger) one_particle.getBid().getValue(i)).getValue());

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;
			}

			// RESAMPLING OF THE STOCHASTIC WITH OMNI DIRECTIONAL AFTER 3
			// EXPLORATION ROUNDS
			if (distrib == "resampled_stoch_neighborhood_omni_end"
					|| distrib == "resampled_stoch_neighborhood_omni_end_corrected_particle"
					|| distrib == "resampled_stoch_neighborhood_omni_end_corrected_particle_reachtop") {

				// RESULTS:
				// @ parameters:
				// particle_count = 1; // p
				// exploration_length_initial = 3; // ex
				// expo_step_size=3; // the first epoch only, after that it gets
				// reduced to 1
				// epoch_to_start_omni=35; // ep

				// Covers a large area very quickly and relatively well. Could
				// be useful.
				// RESULTS:
				// @ parameters:
				// particle_count = 1; // p
				// exploration_length_initial = 3; // ex
				// expo_step_size=3; // the first epoch only, after that it gets
				// reduced to 1
				// epoch_to_start_omni=259; // ep <<----------

				// Choose 1 particle, the next in the round robin queue :

				BidDetails one_particle;
				if (best_past_exploration.size() != 0) {
					one_particle = best_past_exploration.get(counter_neighborhood % best_past_exploration.size());
				} else {
					one_particle = best_past_exploration.get(0);
				}

				// Here we get the component of the direction sampled earlier.
				// Fortunately, this is a perfect ocassion to expand to all
				// directions, by simply taking the "1st quadrant" direction and
				// randomizing the signs of each component.
				// System.out.println(n_bid);

				a = Character.getNumericValue(n_bid.charAt(i - this.min_issues));

				// True OMNI
				if (neighbor_omni)
					a = a - 1;
				if (resample_epoch > 11)
					neighbor_omni = true;
				if (resample_epoch > 15)
					distrib = "resampled_stoch_neighborhood_omni_end_corrected_particle";

				// CUSTOMIZABLE
				// if (resample_epoch>this.epoch_to_start_omni && a==0) a=1; //
				// also flip zeros to 1s
				if (resample_epoch > this.epoch_to_start_omni)
					range_pct = 0.8; // start doing a very tight range (80%), so
										// only the very best bids are
										// considered.
				// if (resample_epoch>this.epoch_to_start_omni)
				// expo_step_size=3; // ENABLE TUNNELING THROUGH WALLS OF width
				// 2 or less.
				if (resample_epoch > this.epoch_to_start_omni)
					expo_step_size = 1; // DISABLE TUNNELING THROUGH WALLS

				// Multiply exploration distance
				// So here this means, how far agent looks in the given
				// direction
				a = a * expo_step_size;

				// Center on zero
				// a=a+0;

				// Center on expected value of option
				// a=(int) Math.round(a+(mi+mx)/2);

				// Center on the particle. What is the component for this issue?
				a = (int) Math.round(a + ((ValueInteger) one_particle.getBid().getValue(i)).getValue());

				// Clip at range
				if (a < mi)
					a = mi;
				if (a > mx)
					a = mx;
			}

			// End of distribution choice. The generated option number should be
			// in 'a'

			// Fill this part of the bid
			bidP.put(i, new ValueInteger(a));
		}
	}

	/**
	 * Check if we've currently reached the Kalai-Somordinsky point estimate. We
	 * take hysteresis and offset into account by adding them to the actual
	 * point (being above the point if either of them is larger then zero)
	 * 
	 * @param hysteresis
	 *            Currently used hysteresis
	 * @param offset
	 *            Currently used offset
	 * @return
	 */
	boolean isKalaiReached(double hysteresis, double offset) {

		return false;

		// Has to be rewritten to access local history, and some blind oppoent
		// model.

		// try {
		// // calculation is only possible if we have done any bid
		// if (this.negotiationSession.getOwnBidHistory().getHistory().size()>0)
		// {
		// // generate new bidspace
		// BidSpace bidspace = new
		// BidSpace(negotiationSession.getUtilitySpace(),
		// opponentModel.getOpponentUtilitySpace(),false);
		//
		// // calculate the kalai util
		// double kalaiutil = this.negotiationSession.getUtilitySpace()
		// .getUtility(bidspace.getKalaiSmorodinsky().getBid());
		//
		// // get the last bid's util
		// double lastbidutil =
		// this.negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil();
		//
		// // return true if last bid was below kalai (taking hysterisis and
		// offset into account)
		// return kalaiutil+hysteresis+offset>=lastbidutil;
		// } else return false; // if no bids in history
		//
		// } catch (Exception e) {
		// return false; // return not reached if calculation can't be made
		// }
	}

	/**
	 * Helper class with functions needed for timing. These are straightforward
	 * statistics calculations.
	 */
	static class SetMath {

		public static double sum(List<Double> input) {
			double sum = 0;
			for (int i = 1; i < input.size(); i++)
				sum += input.get(i) - input.get(i - 1);
			return sum;
		}

		public static double mean(List<Double> input) {
			return sum(input) / input.size();
		}

		public static double variance(List<Double> input) {
			double mean = mean(input);
			// [1/(n-1)] * sum(yi -mean)^2
			double sum = 0;
			double prefix = 1D / (input.size() - 1D);
			for (int i = 1; i < input.size(); i++)
				sum += Math.pow((input.get(i) - input.get(i - 1) - mean), 2);
			return prefix * sum;
		}

		public static double avg(List<Double> input, int n) {
			if (input.size() <= n)
				return 0;
			double sum = 0;
			for (int i = input.size() - n; i < input.size(); i++) {
				sum += input.get(i) - input.get(i - 1);
			}
			return sum / ((double) n);
		}
	}

	public void setup_callbacks() {

		// Sets up the preference model evaluators as callbacks, so that they
		// can be accessed from outside the app, eg. from Matlab
		// matlabLink.enter_callback_functions(this);

		// EvalLink.set_up_listener(this);
		// EvalLink.set_up_listener(new ArrayList<String>(7));
		// kjh=EvalLink.set_up_listener(4);
		// matlabLink.tell_companion_to_link_back_to_this_agent();

		// Now pause and wait for incoming request.
		// Hopefully this will stop the EOF when rounds are exchanged.
		try {
			Thread.sleep(1000 * 770);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("Cannot sleep the thread");
		}

	}

	public double getUtility_Own_model(Integer[] bid) {

		// Translate integer values bid vector into a Genius API Bid

		HashMap<java.lang.Integer, Value> bidP = new HashMap<Integer, Value>();

		for (int j = 1; j <= bid.length; j++) {

			bidP.put(j, new ValueInteger(bid[j - 1].intValue()));

		}

		double actualUtil = 0;

		try {

			Domain dom = negotiationSession.getDomain();
			Bid Formatted_bid;
			Formatted_bid = new Bid(dom, bidP);

			// System.out.println(Formatted_bid.toString());

			actualUtil = utilitySpace.getUtility(Formatted_bid);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return actualUtil;
	}

	public double getUtility_Opponent_model(Integer[] bid) {

		// Translate integer values bid vector into a Genius API Bid

		// HashMap<java.lang.Integer,Value> bidP=new HashMap<Integer, Value>();
		// for (int j = 0; j < bid.length; j++) {
		// bidP.put(j, new ValueInteger(bid[j]));
		// }
		HashMap<java.lang.Integer, Value> bidP = new HashMap<Integer, Value>();

		for (int j = 1; j <= bid.length; j++) {

			bidP.put(j, new ValueInteger(bid[bid.length - (j)].intValue()));

		}

		// Bid Formatted_bid;
		double actualUtil = 0;
		try {

			// Formatted_bid = new Bid(negotiationSession.getDomain(),bidP);
			actualUtil = this.opponentModel.getOpponentUtilitySpace()
					.getUtility(new Bid(negotiationSession.getDomain(), bidP));

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return 0;
	}

	private BidDetails deepcopyBid(BidDetails source) {

		try {
			return arrayAsBidDetails(bidDetailsAsArray(source));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public String getName() {
		return "2014 - BidSpaceExtractor_middle";
	}

}
