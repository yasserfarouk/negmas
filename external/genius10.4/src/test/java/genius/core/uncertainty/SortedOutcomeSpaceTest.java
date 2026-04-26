package genius.core.uncertainty;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.utility.AdditiveUtilitySpace;

public class SortedOutcomeSpaceTest {
	private final static String RESOURCES = "src/test/resources/";
	private static final String DISCRETEDOMAIN = RESOURCES
			+ "partydomain/party_domain.xml";

	@Test
	public void sortedOutcomeSpaceTest() throws Exception {
		Domain domain;
		domain = new DomainImpl(DISCRETEDOMAIN);

		List<Bid> r = new ArrayList<Bid>();
		for (int i = 0; i < 3000; i++) {
			Bid bid = domain.getRandomBid(new Random());
			if (!r.contains(bid))
				r.add(bid);
		}
		BidRanking bidRanking = new BidRanking(r, 0, 1);

		System.out.println(bidRanking);
		AdditiveUtilitySpaceFactory factory = new AdditiveUtilitySpaceFactory(
				domain);
		AdditiveUtilitySpace us = factory.getUtilitySpace();

		factory.estimateUsingBidRanks(bidRanking);

		System.out.println(us);

		System.out.println(us.getUtility(r.get(0)));

		SortedOutcomeSpace outcomeSpace = new SortedOutcomeSpace(us);
		System.out.println(outcomeSpace);

		// int no = 1;
		// ValueDiscrete v0 = (ValueDiscrete) bid.getValue(no);
		// Issue issue = domain.getIssues().get(no);
		// factory.setUtility(issue, v0, .314);

		// TODO there should be assertions here that determine if the tests were
		// running OK
	}

}
