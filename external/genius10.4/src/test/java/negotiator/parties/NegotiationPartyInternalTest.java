package negotiator.parties;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.net.URL;

import org.junit.Test;

import genius.core.AgentID;
import genius.core.Deadline;
import genius.core.exceptions.InstantiateException;
import genius.core.exceptions.NegotiatorException;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.parties.SessionsInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.DomainRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.RepositoryException;
import genius.core.session.Session;
import genius.core.timeline.Timeline;

public class NegotiationPartyInternalTest {
	private static final String PARTY1_UTIL = "file:src/test/resources/partydomain/party1_utility.xml";
	private static final String DOMAIN_REPO = "file:src/test/resources/partydomain/party_domain.xml";

	@Test
	public void createParty() throws InstantiateException, RepositoryException,
			NegotiatorException, IOException {
		PartyRepItem partyRepItem = new PartyRepItem(
				"agents.nastyagent.NullBid");
		DomainRepItem domain = new DomainRepItem(new URL(DOMAIN_REPO));
		ProfileRepItem profileRepItem = new ProfileRepItem(new URL(PARTY1_UTIL),
				domain);

		Session session = mock(Session.class);
		Timeline timeline = mock(Timeline.class);
		Deadline deadline = mock(Deadline.class);
		when(session.getTimeline()).thenReturn(timeline);
		when(session.getDeadlines()).thenReturn(deadline);
		SessionsInfo info = new SessionsInfo(null, PersistentDataType.DISABLED,
				true);
		new NegotiationPartyInternal(partyRepItem, profileRepItem, session,
				info, new AgentID("testname"));

	}

	@Test(expected = NullPointerException.class)
	public void createPartyWithNullID() throws InstantiateException,
			RepositoryException, NegotiatorException, IOException {
		PartyRepItem partyRepItem = new PartyRepItem(
				"agents.nastyagent.NullBid");
		DomainRepItem domain = new DomainRepItem(new URL(DOMAIN_REPO));
		ProfileRepItem profileRepItem = new ProfileRepItem(new URL(PARTY1_UTIL),
				domain);

		Session session = mock(Session.class);
		SessionsInfo info = new SessionsInfo(null, PersistentDataType.DISABLED,
				true);
		new NegotiationPartyInternal(partyRepItem, profileRepItem, session,
				info, null);

	}

}
