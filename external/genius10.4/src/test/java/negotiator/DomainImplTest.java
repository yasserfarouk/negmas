package negotiator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.DomainImpl;

public class DomainImplTest {

	private static final String DISCRETEDOMAIN = "src/test/resources/partydomain/party_domain.xml";
	private static final String INTEGERDOMAIN = "src/test/resources/IntegerDomain/IntegerDomain.xml";
	private static final String REALDOMAIN = "src/test/resources/2nd_hand_car/car_domain.xml";
	private static final String NONLINEARDOMAIN = "src/test/resources/S-1NIKFRT-1/S-1NIKFRT-1-domain.xml";

	private static final String[] testNames = new String[] { DISCRETEDOMAIN, INTEGERDOMAIN, NONLINEARDOMAIN,
			REALDOMAIN };

	@Test
	public void testDefaultConstructor() {
		Domain domain = new DomainImpl();

		assertNotNull(domain.getName());
		assertTrue(domain.getIssues().isEmpty());
		assertNotNull(domain.getObjectivesRoot());
		assertEquals(0, domain.getNumberOfPossibleBids());
		assertEquals(1, domain.getObjectives().size());
		Bid bid = domain.getRandomBid(null);
		assertTrue(bid.getIssues().isEmpty());
		assertTrue(bid.getValues().isEmpty());
	}

	@Test
	public void testLoadFileDiscreteDomain() throws Exception {
		Domain domain = new DomainImpl(DISCRETEDOMAIN);
		assertEquals(DISCRETEDOMAIN, domain.getName());
		assertEquals(6, domain.getIssues().size());
		assertNotNull(domain.getObjectivesRoot());
		assertEquals(3072, domain.getNumberOfPossibleBids());
		// root objective adds 1 to the number of issues.
		assertEquals(7, domain.getObjectives().size());
		Bid bid = domain.getRandomBid(null);
		assertEquals(6, bid.getIssues().size());
		assertEquals(6, bid.getValues().size());

	}

	@Test
	public void testLoadFileIntegerDomain() throws Exception {
		Domain domain = new DomainImpl(INTEGERDOMAIN);
		assertEquals(INTEGERDOMAIN, domain.getName());
		assertEquals(2, domain.getIssues().size());
		assertNotNull(domain.getObjectivesRoot());
		assertEquals(121, domain.getNumberOfPossibleBids());
		// root objective adds 1 to the number of issues.
		assertEquals(3, domain.getObjectives().size());
		Bid bid = domain.getRandomBid(null);
		assertEquals(2, bid.getIssues().size());
		assertEquals(2, bid.getValues().size());

	}

	@Test
	public void testLoadFileRealDomain() throws Exception {
		Domain domain = new DomainImpl(REALDOMAIN);
		assertEquals(REALDOMAIN, domain.getName());
		assertEquals(5, domain.getIssues().size());
		assertNotNull(domain.getObjectivesRoot());
		assertEquals(13125, domain.getNumberOfPossibleBids());
		// root objective adds 1 to the number of issues.
		assertEquals(6, domain.getObjectives().size());
		Bid bid = domain.getRandomBid(null);
		assertEquals(5, bid.getIssues().size());
		assertEquals(5, bid.getValues().size());

	}

	@Test
	public void testLoadFileNonlinearDomain() throws Exception {
		Domain domain = new DomainImpl(NONLINEARDOMAIN);
		assertEquals(NONLINEARDOMAIN, domain.getName());
		assertEquals(10, domain.getIssues().size());
		assertNotNull(domain.getObjectivesRoot());
		assertEquals(10000000000l, domain.getNumberOfPossibleBids());
		// root objective adds 1 to the number of issues.
		assertEquals(11, domain.getObjectives().size());
		Bid bid = domain.getRandomBid(null);
		assertEquals(10, bid.getIssues().size());
		assertEquals(10, bid.getValues().size());

	}

	@Test
	public void testEquals() throws Exception {
		DomainImpl emptydomain = new DomainImpl();
		for (String filename : testNames) {
			Domain domain = new DomainImpl(filename);
			Domain domain2 = new DomainImpl(filename);
			assertEquals(domain, domain);
			assertEquals(domain, domain2);
			assertFalse(domain.equals(null));
			assertFalse(domain.equals(1));
			assertFalse(domain.equals(emptydomain));
		}

	}

}
