package agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.misc.Range;
 

/**
 * Just repeats the top 25 best bids. Best bids are defined as bids by utilrange (currently 0.9 to 1.0). The point of
 * this is to give our agent some time to model the opponent
 */
public class OpeningGameStrategy extends AbstractBiddingStrategy
{
  

	private BidSpaceExtractor_opening bidSpaceExtractor;
	
	OpeningGameStrategy(NegotiationSession negotiationSession, OpponentModel opponentModel) 
	{
		super(negotiationSession, opponentModel);
		
		// NONLINEAR modif
		bidSpaceExtractor=new BidSpaceExtractor_opening();
		try {
			bidSpaceExtractor.init(negotiationSession, opponentModel, null, null);
			  
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	 
	}

	@Override // see parent
	Bid getBid() 
	{
		// NONLINEAR modification
		//  Just offer the bids that is only good for us
			return bidSpaceExtractor.determineNextBid().getBid();			
	  
	}
 
}
