package agents.anac.y2019.harddealer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;

public class HardDealer_OM extends OpponentModel {

	private double learnCoeff;

	private double issueUpdate;

	private int valueUpdate;
	
	private double minOppUtil = 0;             // Minimum utility value for opponent when Negotiation Time ends 
	private double maxOppUtil = 1;			   // Maximum utility value for opponent when Negotiation Time starts
	private double nOfHypothesis;
	private double nOfIssues;
	HashMap<List<Issue>, Double>  probHyp = new HashMap<List<Issue>, Double>();	// the probabilities for each hypothesis
	List<List<Issue>> spaceOfHypothesis = new ArrayList<List<Issue>>(); // All the possible hypothesis
	HashMap<List<Issue>, Double> oppUF = new HashMap<List<Issue>, Double>(); // List of utility evaluations for each hypothesis
	HashMap<List<Issue>, Double> probHypGivenBid = new HashMap<List<Issue>, Double>(); // List of probabilities for each hypothesis given as bid
	List<Double> oppTargetUtility = new ArrayList<Double>(); // List of target utilities for the opponent for each time interval
	HashMap<List<Issue>,HashMap<Integer, Double>> ListWeight = new HashMap<List<Issue>, HashMap<Integer, Double>>();
	
	HashMap<Objective, Double> simpleWeights = new HashMap<Objective, Double>(); // List of weights for the SimpleLearning model
	
	TimeLineInfo negTime;

	public void init(NegotiationSession negotiationSession,
			Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		if (parameters != null && parameters.get("l") != null) {
			learnCoeff = parameters.get("l");
		} else {
			learnCoeff = 0.13;
		}
		negTime = negotiationSession.getTimeline();
		valueUpdate = 1;
		opponentUtilitySpace = (AdditiveUtilitySpace) negotiationSession
				.getUtilitySpace().copy();
		nOfIssues = opponentUtilitySpace.getDomain().getIssues().size();
		/*
		 * This is the value to be added to weights of unchanged issues before
		 * normalization. Also the value that is taken as the minimum possible
		 * weight, (therefore defining the maximum possible also).
		 */
		issueUpdate = learnCoeff / nOfIssues;

		initializeModel();

	}
	
	/**
	 * Update both the bayesian learning model as the simple learning model
	 */

	@Override
	public void updateModel(Bid bid, double time) {
		double alpha = 1;
		// Calculate target utility for the opponent given the concession formula
		double targetUtility = maxOppUtil - (maxOppUtil - minOppUtil) * 
							Math.pow((negTime.getCurrentTime() / negTime.getTotalTime()), alpha);
		
		oppTargetUtility.add(targetUtility);
		
		/**
		 * For each hypothesis, calculate the utility of the current bid given the weights of that hypothesis
		 * Calculation of utility of bid inspired by getUtility() function in the AdditiveUtilitySpace of genius
		 */
		for (List<Issue> Hypothesis : spaceOfHypothesis) {
			HashMap<Integer, Double> Weights = ListWeight.get(Hypothesis);
			HashMap<Integer, Value> Values = bid.getValues();
			double CompleteUF = 0;
			for (Issue issues : Hypothesis) {
				for (Entry<Objective, Evaluator> e : opponentUtilitySpace.getEvaluators()) {
					EvaluatorDiscrete evaluator = (EvaluatorDiscrete) e.getValue();
					double value = 0;
					try {
						value = evaluator.getEvaluation((ValueDiscrete)Values.get(issues.getNumber()));
						double OppUtilityFun = value * Weights.get(issues.getNumber());
						CompleteUF = CompleteUF + OppUtilityFun;
					} catch (Exception e1) {

					}
				}
			}
			if(CompleteUF > 1) {
				CompleteUF = 1;
			}
			oppUF.put(Hypothesis, CompleteUF);
		}
		
		/**
		 * Calculate the probability of the hypothesis given this bid as the 1 - the distance between the calculated utility and the target utility
		 */
		for (List<Issue> Hypothesis : spaceOfHypothesis) {
			probHypGivenBid.put(Hypothesis, 1 - Math.abs(oppUF.get(Hypothesis) - oppTargetUtility.get(oppTargetUtility.size()-1)));
		}
		
		double sumProbHypGivenBid = 0;
		for (List<Issue> hypothesis : spaceOfHypothesis) {
			sumProbHypGivenBid += probHypGivenBid.get(hypothesis) * probHyp.get(hypothesis);
		}
		
		double maxProb = 0;
		List<Issue> maxHyp = new ArrayList<>();
		// Bayesian Rule to calculate the new probability of each hypothesis
		for (List<Issue> hypothesis : spaceOfHypothesis) {
			double prob = probHypGivenBid.get(hypothesis) * probHyp.get(hypothesis) / sumProbHypGivenBid;
			 if(Double.isNaN(prob))
				 prob = 0;
			probHyp.put(hypothesis, prob);
			// Record which hypothesis has the highest probability to use in the actual weights for the model
			if(prob > maxProb) {
				maxProb = prob;
				maxHyp = hypothesis;
			}
		}
		
		// Only start updating the simple learning and updating weights and values after at least 2 opponent bids
		if (negotiationSession.getOpponentBidHistory().size() < 2) {
			return;
		}
		/**
		 * Gets the last 5 bids, or if there are less than 5 bids, return all the last bids.
		 */
		ArrayList<BidDetails> multipleBids = new ArrayList<BidDetails>();
		BidDetails oppBid = negotiationSession.getOpponentBidHistory().getHistory()
					.get(negotiationSession.getOpponentBidHistory().size() - 1);
		multipleBids.add(oppBid);
		BidDetails prevOppBid1 = negotiationSession.getOpponentBidHistory().getHistory()
					.get(negotiationSession.getOpponentBidHistory().size() - 2);
		multipleBids.add(prevOppBid1);
		if (negotiationSession.getOpponentBidHistory().size() > 2) {
			BidDetails prevOppBid2 = negotiationSession.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory().size() - 3);
			multipleBids.add(prevOppBid2);
		}
		if (negotiationSession.getOpponentBidHistory().size() > 3) {
			BidDetails prevOppBid3 = negotiationSession.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory().size() - 4);
			multipleBids.add(prevOppBid3);
		}
		if (negotiationSession.getOpponentBidHistory().size() > 4) {
			BidDetails prevOppBid4 = negotiationSession.getOpponentBidHistory()
					.getHistory()
					.get(negotiationSession.getOpponentBidHistory().size() - 5);
			multipleBids.add(prevOppBid4);
		}
		/**
		 * Get for each of the issues how many bids the value has not changed
		 */
		HashMap<Integer, Integer> lastDiffSet = last5oppBid(multipleBids);

		/**
		 * Calculate the new total weight
		 * Each weight will have the issue Update value * the number of bids the value in this issue has not changed added to it
		 */
		double totalweight = 0;
		for (Integer i : lastDiffSet.keySet()) {
			double weight = opponentUtilitySpace.getWeight(i);
			if (lastDiffSet.get(i) == 4) {
				totalweight += (weight + 4 * issueUpdate);
			}
			else if (lastDiffSet.get(i) == 3) {
				totalweight += (weight + 3 * issueUpdate);
			}
			else if (lastDiffSet.get(i) == 2) {
				totalweight += (weight + 2 * issueUpdate);
			}
			else if (lastDiffSet.get(i) == 1) {
				totalweight += (weight + 1 * issueUpdate);
			}
			else {
				totalweight += weight;
			}
		}

		/**
		 * Update the weights to the new weights as explained above, but now normalized corresponding to the total weight 
		 */
		for (Integer i : lastDiffSet.keySet()) {
			Objective issue = opponentUtilitySpace.getDomain()
			.getObjectivesRoot().getObjective(i);
			double weight = opponentUtilitySpace.getWeight(i);
			double newWeight;
			if (lastDiffSet.get(i) == 4) {
				newWeight = (weight + 4 * issueUpdate) / totalweight;
			}
			else if (lastDiffSet.get(i) == 3) {
				newWeight = (weight + 3 * issueUpdate) / totalweight;
			}
			else if (lastDiffSet.get(i) == 2) {
				newWeight = (weight + 2 * issueUpdate) / totalweight;
			}
			else if (lastDiffSet.get(i) == 1) {
				newWeight = (weight + 1 * issueUpdate) / totalweight;
			}
			else {
				newWeight = weight / totalweight;
			}
			simpleWeights.put(issue, newWeight);
		}		

		/*
		 * The weights for the opponent model are updated to a combination of the bayesian model and the simple learning model
		 * The weights are calculated as the probability of the highest hypothesis * the weight corresponding to this hypothesis
		 * Combined with 1 - that probability * the weights as calculated by the simple learning model
		 */
		HashMap<Integer, Double> maxWeights = ListWeight.get(maxHyp);
		for(Issue issue: maxHyp) {
			opponentUtilitySpace.setWeight(issue, (maxProb * maxWeights.get(issue.getNumber())) + ((1- maxProb) * simpleWeights.get( (Objective) issue) ) );
		}

		
		try {
			for (Entry<Objective, Evaluator> e : opponentUtilitySpace
					.getEvaluators()) {
				EvaluatorDiscrete value = (EvaluatorDiscrete) e.getValue();
				IssueDiscrete issue = ((IssueDiscrete) e.getKey());
				/*
				 * add constant learnValueAddition to the current preference of
				 * the value to make it more important
				 */
				ValueDiscrete issuevalue = (ValueDiscrete) multipleBids.get(0).getBid()
						.getValue(issue.getNumber());
				Integer eval = value.getEvaluationNotNormalized(issuevalue);
				value.setEvaluation(issuevalue, (valueUpdate + eval));
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public double getBidEvaluation(Bid bid) {
		double result = 0;
		try {
			result = opponentUtilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	public String getName() {
		return "HardDealer_OM";
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("l", 0.13,
				"The learning coefficient determines how quickly the issue weights are learned"));
		return set;
	}

	/**
	 * Initialize both the simple learning and bayesian learning model
	 */
	private void initializeModel() {
		double commonWeight = 1D / nOfIssues;

		for (Entry<Objective, Evaluator> e : opponentUtilitySpace
				.getEvaluators()) {

			opponentUtilitySpace.unlock(e.getKey());
			// Set the weights to 1 / number of issues
			e.getValue().setWeight(commonWeight);
			try {
				// set all value weights to one (they are normalized when
				// calculating the utility)
				for (ValueDiscrete vd : ((IssueDiscrete) e.getKey())
						.getValues())
					((EvaluatorDiscrete) e.getValue()).setEvaluation(vd, 1);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
		
		List<Issue> issues = negotiationSession.getIssues();
		// Initialise the space of hypothesis
		generateHypSpace(issues);
		
		nOfHypothesis = spaceOfHypothesis.size();
		
		// Initialise the probabilities of each hypothesis to 1 / number of hypothesis
		for (List<Issue> hypothesis : spaceOfHypothesis) {          
			probHyp.put(hypothesis, 1D / nOfHypothesis);
		}
		
		
		for (List<Issue> Hypothesis : spaceOfHypothesis) {
			HashMap<Integer, Double> hypWeights = new HashMap<Integer, Double>();
			for (Issue issue : Hypothesis) {
				// The weights for each issue in a hypothesis is defined as 2 * rank / number of issues * number of issues - 1
				double OppNewWeight = 2 * (Hypothesis.indexOf(issue) + 1) / (nOfIssues * (nOfIssues + 1));
				hypWeights.put(issue.getNumber(), OppNewWeight);
			}
			ListWeight.put(Hypothesis, hypWeights);
			oppUF.put(Hypothesis, 0D);
		}
	}

	/*
	 * Returns for each issue how many bids in a row the value for that issue has not changed
	 */
	private HashMap<Integer, Integer> last5oppBid(ArrayList<BidDetails> multipleBids) {

		HashMap<Integer, Integer> oppBiddiff = new HashMap<Integer, Integer>();
		try {
			for (Issue i : opponentUtilitySpace.getDomain().getIssues()) {
				Value value1 = null;
				Value value2 = null;
				Value value3 = null;
				Value value4 = null;
				Value value5 = null;
				value1 = multipleBids.get(0).getBid().getValue(i.getNumber());
				value2 = multipleBids.get(1).getBid().getValue(i.getNumber());
				if (multipleBids.size() > 2) {
					value3 = multipleBids.get(2).getBid().getValue(i.getNumber());
				}
				if (multipleBids.size() > 3) {
					value4 = multipleBids.get(3).getBid().getValue(i.getNumber());
				}
				if (multipleBids.size() > 4) {
					value5 = multipleBids.get(4).getBid().getValue(i.getNumber());
				}
				/**
				 * Checks first if the last 4 values equals the current value
				 * If not, it checks for 3, 2 and 1 last values.
				 */
				if (value1.equals(value2) && value2.equals(value3) && value3.equals(value4) && value4.equals(value5)) {
					oppBiddiff.put(i.getNumber(), 4);
				}
				else if (value1.equals(value2) && value2.equals(value3) && value3.equals(value4) && !(value4.equals(value5))) {
					oppBiddiff.put(i.getNumber(), 3);
				}
				else if (value1.equals(value2) && value2.equals(value3) && !(value3.equals(value4))) {
					oppBiddiff.put(i.getNumber(), 2);
				}
				else if (value1.equals(value2) && !(value2.equals(value3))) {
					oppBiddiff.put(i.getNumber(), 1);
				}
				else if (!(value1.equals(value2))) {
					oppBiddiff.put(i.getNumber(), 0);
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		return oppBiddiff;
	}
	
	
	/*
	 * Creates the list of possible hypothesis by creating all permutations of the list of issues
	 * All the following code for creating permutations was inspired by https://stackoverflow.com/questions/36373719/java-permutations-of-an-array
	 */
	private ArrayList<List<Issue>> generateHypSpace(List<Issue> issues) {	
		return permutations(issues);		
	}
		

	/*
	 * Swaps two issues in a hypothesis
	 */
    private List<Issue> swap(List<Issue> spaceOfHypothesis2, int i, int j) {
    	Issue tmp = spaceOfHypothesis2.get(i);
    	spaceOfHypothesis2.set(i, spaceOfHypothesis2.get(j));
    	spaceOfHypothesis2.set(j, tmp);
    	return spaceOfHypothesis2;
    }

    /*
     * Creates a list of all permutations of a list of issues
     */
    private void permutations(List<Issue> spaceOfHypothesis2, int loc, int len) {
    	// If you reach the end of the list, this permutation is done
    	// Make a hard copy of this permutation to prevent referencing issues and add it to the result
        if (loc == len){
        	ArrayList<Issue> copy = new ArrayList<Issue>();
        	for(Issue i : spaceOfHypothesis2) {
        		copy.add(i);
        	}
            spaceOfHypothesis.add(copy);
            return;
        }

        // Make all permutations from the next issue
        permutations(spaceOfHypothesis2, loc + 1, len);
        for (int i = loc + 1; i < len; i++) {
            // Swap the current issue with the issue at index i
            spaceOfHypothesis2 = swap(spaceOfHypothesis2, loc, i);
            // Create all permutations with these two issues swapped
            permutations(spaceOfHypothesis2, loc + 1, len);
            // Restore the permutation
            spaceOfHypothesis2 = swap(spaceOfHypothesis2, loc, i);
        }
    }

    /*
     * Create the permutations by intialising the result, and starting at index 0
     */
    public ArrayList<List<Issue>> permutations(List<Issue> spaceOfHypothesis2) {
        ArrayList<List<Issue>> result = new ArrayList<List<Issue>>();
        permutations(spaceOfHypothesis2, 0, spaceOfHypothesis2.size());
        return result;
    }
}
