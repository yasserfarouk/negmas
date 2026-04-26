package genius.core.uncertainty;

import static org.junit.Assert.*;

import java.util.List;

import org.junit.Test;

import genius.core.Bid;
import genius.core.DomainImpl;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.utility.UncertainAdditiveUtilitySpace;
import genius.core.uncertainty.UncertainPreferenceContainer;
import genius.core.uncertainty.UNCERTAINTYTYPE;

public class UserTest {

	String PARTY = "src/test/resources/partydomain/";
	 
	@Test
	public void testElicitBidAndAll() throws Exception {
		
		/**
		 * We start by creating a user and a user model for the party1_utility space.
		 */
		DomainImpl domain = new DomainImpl(PARTY + "party_domain.xml");
		UncertainAdditiveUtilitySpace utilSpace = new UncertainAdditiveUtilitySpace(domain,
				PARTY + "party1_utility.xml");
		User user = new User(utilSpace);
		UncertainPreferenceContainer u = new UncertainPreferenceContainer(utilSpace, UNCERTAINTYTYPE.PAIRWISECOMP);
		UserModel userModel = u.getPairwiseCompUserModel();
		BidRanking bidRank = userModel.getBidRanking();
		
		/**
		 * Here, we begin the testing of the core function elicitBid.
		 */
		int originalSize = bidRank.getSize();
		Bid maxBid = bidRank.getMaximalBid();
		UserModel updated = user.elicitRank(maxBid, userModel);
		
		//Should stay the same because maxBid is already in userModel.
		assertEquals(updated, userModel);
		
		/**
		 * Obtain a random bid not in the user model and add it to updated.
		 * We first test that it was indeed added.
		 * Then we test that it is at the right place in the ranking.
		 */
		Bid bid = maxBid;
		while(bidRank.getBidOrder().contains(bid)) 
			bid = domain.getRandomBid(null);
		updated = user.elicitRank(bid, userModel);
		assertEquals((originalSize + 1), updated.getBidRanking().getSize());
		int bidIndex = updated.getBidRanking().indexOf(bid);
		Bid prevBid = updated.getBidRanking().getBidOrder().get(bidIndex-1);
		Bid nextBid = updated.getBidRanking().getBidOrder().get(bidIndex+1);
		BidDetails prev = new BidDetails(prevBid, utilSpace.getUtility(prevBid));
		BidDetails next = new BidDetails(nextBid, utilSpace.getUtility(nextBid));
		BidDetails newBid = new BidDetails(bid, utilSpace.getUtility(bid));
		int comparResult = (new BidDetailsSorterUtility()).compare(prev, newBid);
		assertEquals(comparResult, 1);
		comparResult = (new BidDetailsSorterUtility()).compare(newBid, next);
		assertEquals(comparResult,1);
		
		/**
		 * Now we check that the updated ranking without new bid is the same as the old ranking.
		 */
		List<Bid> updatedWithoutBid = updated.getBidRanking().getBidOrder();
		updatedWithoutBid.remove(bidIndex);
		assertEquals(updatedWithoutBid, bidRank.getBidOrder());
		
		/**
		 * We finally test that the total bother was well updated. 
		 */
		double elicitationCost = utilSpace.getElicitationCost();
		assertEquals(elicitationCost,user.getTotalBother(),0.00000000001);
	
	}
	
}
