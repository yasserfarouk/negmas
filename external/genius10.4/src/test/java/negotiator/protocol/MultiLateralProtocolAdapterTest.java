package negotiator.protocol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import genius.core.parties.NegotiationParty;
import genius.core.protocol.DefaultMultilateralProtocol;
import genius.core.session.Session;

public class MultiLateralProtocolAdapterTest {

	private DefaultMultilateralProtocol protocol;

	Session session = mock(Session.class);

	@Before
	public void init() {

		protocol = mock(DefaultMultilateralProtocol.class,
				Mockito.CALLS_REAL_METHODS);
	}

	@Test
	public void testEndNego() {
		protocol.endNegotiation();
		assertTrue(protocol.isFinished(session, null));
	}

	@Test
	public void testFilterEmptyList() {
		List<NegotiationParty> negotiationParties = new ArrayList<NegotiationParty>();
		Collection<NegotiationParty> filtered = protocol.includeOnly(
				negotiationParties, NegotiationParty.class);
		assertTrue("filter result not empty", filtered.isEmpty());
	}

	@Test
	public void testFilterOneParty() {
		List<NegotiationParty> negotiationParties = new ArrayList<NegotiationParty>();
		NegotiationParty oneParty = mock(NegotiationParty.class);
		negotiationParties.add(oneParty);

		Collection<NegotiationParty> filtered = protocol.includeOnly(
				negotiationParties, oneParty.getClass());
		assertFalse("filter result is empty", filtered.isEmpty());
	}

	/**
	 * Filter two parties, only first one should remain.
	 */
	@Test
	public void testFilterTwoParties() {
		List<NegotiationParty> negotiationParties = new ArrayList<NegotiationParty>();
		NegotiationParty party1 = mock(NegoParty1.class);
		NegotiationParty party2 = mock(NegoParty2.class);

		negotiationParties.add(party1);
		negotiationParties.add(party2);

		Collection<NegotiationParty> filtered = protocol.includeOnly(
				negotiationParties, party1.getClass());
		assertEquals(1, filtered.size());
		assertEquals(party1, filtered.iterator().next());
	}

	/**
	 * Filter two parties, only first one should remain.
	 */
	@Test
	public void testFilterTwoPartiesExclude() {
		List<NegotiationParty> negotiationParties = new ArrayList<NegotiationParty>();
		NegotiationParty party1 = mock(NegoParty1.class);
		NegotiationParty party2 = mock(NegoParty2.class);

		negotiationParties.add(party1);
		negotiationParties.add(party2);

		Collection<NegotiationParty> filtered = protocol.exclude(
				negotiationParties, party1.getClass());
		assertEquals(1, filtered.size());
		assertEquals(party2, filtered.iterator().next());
	}

	interface NegoParty1 extends NegotiationParty {
	}

	interface NegoParty2 extends NegotiationParty {

	}

}
