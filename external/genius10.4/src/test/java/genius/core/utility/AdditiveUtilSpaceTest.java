package genius.core.utility;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.junit.Test;

import genius.core.DomainImpl;

public class AdditiveUtilSpaceTest {

	String PARTY = "src/test/resources/partydomain/";

	@Test
	public void testAdditive1() throws IOException {
		DomainImpl domain = new DomainImpl(PARTY + "party_domain.xml");
		AdditiveUtilitySpace additive = new AdditiveUtilitySpace(domain,
				PARTY + "party1_utility.xml");

		assertEquals(UTILITYSPACETYPE.LINEAR, UTILITYSPACETYPE
				.getUtilitySpaceType(PARTY + "party1_utility.xml"));
		assertEquals(domain, additive.getDomain());
		assertEquals(6, additive.getNrOfEvaluators());
	}

}
