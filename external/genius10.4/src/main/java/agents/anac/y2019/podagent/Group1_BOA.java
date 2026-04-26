package agents.anac.y2019.podagent;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.*;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.DiscreteTimeline;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.utility.AbstractUtilitySpace;
import negotiator.boaframework.omstrategy.NullStrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;


@SuppressWarnings("serial")
public class Group1_BOA extends BoaParty {
	@Override
	public void init(NegotiationInfo info) {
		// The choice for each component is made here
		AcceptanceStrategy ac = new Group1_AS();
		OfferingStrategy os = new Group1_BS();
		OpponentModel om = new Group1_OM();
		// OMS is part of the opponent model in our agent
		OMStrategy oms = new NullStrategy();

		// All component parameters can be set below.
		Map<String, Double> noparams = Collections.emptyMap();

		// Initialize all the components of this party to the choices defined above
		configure(ac, noparams, os, noparams, om, noparams, oms, noparams);
		super.init(info);
	}

	/*
	 * Estimate utility function in case of preference uncertainty.
	 * 
	 * @see genius.core.parties.AbstractNegotiationParty#estimateUtilitySpace()
	 */
	@Override
	public AbstractUtilitySpace estimateUtilitySpace() {
		AdditiveUtilitySpaceFactory factory = new AdditiveUtilitySpaceFactory(getDomain());
		// Order of bids from low to high
		List<Bid> bidOrder = userModel.getBidRanking().getBidOrder();
		ArrayList<Bid> bidOrderArrayList = new ArrayList(bidOrder);
		
		//Assign utilities to bids that are not in the specified bidOrder
//		for(BidDetails bid: outcomeSpace.getAllOutcomes()) {
//			if(bidOrderArrayList.contains(bid.getBid()))
//				continue;
//			
//			int bestPosition = 0;
//			double bestDistanceTop = 0;
//			double bestDistanceBottom = 0;
//
//			for(int i = 0; i < bidOrderArrayList.size() - 1; i++ ) {
//				Bid higherBid = bidOrderArrayList.get(i);
//				Bid lowerBid = bidOrderArrayList.get(i + 1);
//				double higher = getBidDistance(bid.getBid(), higherBid);
//				double lower = getBidDistance(bid.getBid(), lowerBid);
//				if(higher > bestDistanceTop && lower > bestDistanceBottom) {
//					bestDistanceTop = lower;
//					bestDistanceBottom = higher;
//					bestPosition = i;
//				}
//			}
//			bidOrderArrayList.add(bestPosition + 1, bid.getBid());
//		}
		Random r = new Random();
		
		//Assign utilities to bids that are not in the specified bidOrder
		for(int j = bidOrder.size(); j < 100; j++) {
			Bid bid = factory.getDomain().getRandomBid(r);
			if(bidOrderArrayList.contains(bid))
				continue;
			
			int bestPosition = 0;
			double bestDistanceTop = 0;
			double bestDistanceBottom = 0;

			for(int i = 0; i < bidOrderArrayList.size() - 1; i++ ) {
				Bid higherBid = bidOrderArrayList.get(i);
				Bid lowerBid = bidOrderArrayList.get(i + 1);
				double higher = getBidDistance(bid, higherBid);
				double lower = getBidDistance(bid, lowerBid);
				if(higher > bestDistanceTop && lower > bestDistanceBottom) {
					bestDistanceTop = lower;
					bestDistanceBottom = higher;
					bestPosition = i;
				}
			}
			bidOrderArrayList.add(bestPosition + 1, bid);
		}
		
		
		// Utility of highest and lowest bid
		double low = userModel.getBidRanking().getLowUtility();
		double high = userModel.getBidRanking().getHighUtility();
		List<IssueDiscrete> issues = factory.getIssues();
		// Initialize issues and all weights with 0
		for (IssueDiscrete i : issues) {
			factory.setWeight(i, 1.0 / (double) issues.size());
			for (ValueDiscrete v : i.getValues()) {
				factory.setUtility(i, v, 0.0);
			}
		}
		// Count the number of occurrences of each value
		HashMap<String, Integer> valCounter = new HashMap<String, Integer>();
		for (int i = 0; i < bidOrderArrayList.size(); i++) {
			// Estimate utility of bids in between high and low depending on their position
			// in the ordering
			// Position of bid correlates to it's utility
			// Linear utility function
			// double currentUtil = low + (high-low)*i/bidOrder.size();
			// Exponential utility function that shifts the distribution of utility values
			// towards the higher end of the ordering
			double currentUtil = low + (high - low) * Math.pow(i / (double) bidOrderArrayList.size(), 2);
			Bid currentBid = bidOrderArrayList.get(i);
			// Add the estimated utility of the bid to its values
			for (IssueDiscrete issue : issues) {
				int no = issue.getNumber();
				ValueDiscrete v = (ValueDiscrete) currentBid.getValue(no);
				// Count occurrences
				valCounter.put(v.getValue(), valCounter.getOrDefault(v.getValue(), 0) + 1);
				double oldUtil = factory.getUtility(issue, v);
				factory.setUtility(issue, v, oldUtil + currentUtil);
			}
		}
		ArrayList<Double> issueWeights = new ArrayList<Double>();

		for (IssueDiscrete i : issues) {
			// Calculate the variance of values in the issue
			ArrayList<Integer> weights = new ArrayList<Integer>();
			// Average value weights by taking their average utility
			double avg = 0;
			double counter = 0;
			for (ValueDiscrete v : i.getValues()) {
				factory.setUtility(i, v, factory.getUtility(i, v) / (double) valCounter.getOrDefault(v.getValue(), 0));
				weights.add(valCounter.getOrDefault(v.getValue(), 0));
				if (factory.getUtility(i, v) != 0.0) {
					avg += factory.getUtility(i, v);
					counter++;
				}
			}
			// Square numbers of value occurrences to estimate how varied the responses are
			double var = 0;
			for (int w : weights) {
				var += Math.pow(w, 2);
			}
			issueWeights.add(var);
			
			avg = avg / counter;
			// set 0 values to the average
			for (ValueDiscrete v : i.getValues()) {
				if (factory.getUtility(i, v) == 0.0) {
					factory.setUtility(i, v, avg);
				}
			}
		}
		// Invert and normalize weights
		double sum = 0;
		for (int i = 0; i < issueWeights.size(); i++) {
			// More variance = less utility
			issueWeights.set(i, 1 - issueWeights.get(i));
			sum += issueWeights.get(i);
		}
		// Normalize the weights and set the issue weight
		for (int i = 0; i < issueWeights.size(); i++) {
			issueWeights.set(i, issueWeights.get(i) / sum);
			factory.setWeight(issues.get(i), issueWeights.get(i));
		}
		factory.normalizeWeights();
		return factory.getUtilitySpace();
	}

	@Override
	public String getDescription() {
		return "Group1_BOA";
	}

	public double getBidDistance(Bid bid1, Bid bid2) {
		HashSet<Value> merged = new HashSet(bid1.getValues().values());
		merged.addAll(bid2.getValues().values());
		return bid1.countEqualValues(bid2) / merged.size();
	}

}
