package agents.anac.y2019.harddealer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This implementation is based on the code provided in the Boaexamples directory. Specifically the code of BestBid has been used and altered.
 * @author svenhendrikx
 *
 */
public class HardDealer_OMS extends OMStrategy {

	double updateThreshold = 1.1;

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, Map<String, Double> parameters) {
		super.init(negotiationSession, model, parameters);
		if (parameters.get("t") != null) {
			updateThreshold = parameters.get("t").doubleValue();
		} else {
			System.out.println("OMStrategy assumed t = 1.1");
		}
	}

	@Override
	public BidDetails getBid(List<BidDetails> allBids) {
		if (allBids.size() == 1) {
			return allBids.get(0);
		}
		
		// Check if the model is working, i.e. not returning only zeroes.
		boolean allWereZero = true;
		
		// Will contain x best bids
		List<BidDetails> bestbids = new ArrayList<BidDetails>();
		for (BidDetails bid : allBids) {
			
			double evaluation = model.getBidEvaluation(bid.getBid());
			
			// model works
			if (evaluation > 0.0001) 
			{
				allWereZero = false;
			}
			
			// bestbids's allowed size will decrease during negotiation. For more info, see OMstrategy part in report
			if (bestbids.size() < (int)(5 * (1 - negotiationSession.getTime())) + 1)
			{
				bestbids.add(bid);
			}
			else
			{
				// Find the five best bids according to the opponent model
				bestbids.sort(new SortBidsOpponent(model));
			
				if (model.getBidEvaluation(bestbids.get(0).getBid()) < evaluation)
					bestbids.set(0, bid);
			}
			
		}
		
		// 4. The opponent model did not work, therefore, offer a random bid.
		if (allWereZero) {
			Random r = new Random();
			return allBids.get(r.nextInt(allBids.size()));
		}
		
		
		bestbids.sort(new SortBids());
		// Find the best bid from the bids the opponent model selected.
		BidDetails myBid = bestbids.get(bestbids.size() - 1);
		
		if (negotiationSession.getOpponentBidHistory().isEmpty())
		{
			return myBid;
		}
		else
		{
			// If the opponent has offered something before with a better utility than our current selected bid, return their best bid.
			BidDetails theirBestBid = negotiationSession.getOpponentBidHistory().getBestBidDetails();
			
			if (myBid.getMyUndiscountedUtil() >= theirBestBid.getMyUndiscountedUtil())
			{
				return myBid;
			}
			else
			{
				return theirBestBid;
			}
		}
		
		
	}
	
	@Override
	public boolean canUpdateOM() {
		return negotiationSession.getTime() < updateThreshold;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("t", 1.1, "Time after which the OM should not be updated"));
		return set;
	}

	@Override
	public String getName() {
		return "HardDealer_OMS";
	}

}
