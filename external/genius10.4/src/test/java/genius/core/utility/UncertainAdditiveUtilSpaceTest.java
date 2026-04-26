package genius.core.utility;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.junit.Test;

import genius.core.DomainImpl;

public class UncertainAdditiveUtilSpaceTest {

	String PARTY = "src/test/resources/partydomain/";

	@Test
	public void testAdditive1() throws IOException {
		DomainImpl domain = new DomainImpl(PARTY + "party_domain.xml");
		UncertainAdditiveUtilitySpace additive = new UncertainAdditiveUtilitySpace(
				domain, PARTY + "party1_utility_uncertain.xml");

		assertEquals(UTILITYSPACETYPE.UNCERTAIN, UTILITYSPACETYPE
				.getUtilitySpaceType(PARTY + "party1_utility_uncertain.xml"));
		assertEquals(domain, additive.getDomain());
		assertEquals(6, additive.getNrOfEvaluators());
		assertEquals((Integer) 1, additive.getComparisons());
		assertEquals((Integer) 2, additive.getErrors());
		assertEquals(false, additive.isExperimental());
	}

	@Test
	public void testUncertain2() throws IOException {
		DomainImpl domain = new DomainImpl(PARTY + "party_domain.xml");
		UncertainAdditiveUtilitySpace additive = new UncertainAdditiveUtilitySpace(
				domain, PARTY + "party2_utility_uncertain.xml");
		assertEquals(UncertainAdditiveUtilitySpace.SEED, additive.getSeed());
	}

}
