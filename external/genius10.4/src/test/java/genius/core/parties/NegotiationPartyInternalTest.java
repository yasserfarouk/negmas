package genius.core.parties;

import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.Test;

import genius.core.AgentID;
import genius.core.Deadline;
import genius.core.Domain;
import genius.core.exceptions.InstantiateException;
import genius.core.exceptions.NegotiatorException;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.RepositoryException;
import genius.core.session.Session;
import genius.core.timeline.Timeline;
import genius.core.utility.UncertainAdditiveUtilitySpace;

public class NegotiationPartyInternalTest {
	/**
	 * Test that NegotiationPartyInternal recognises the utilityspace is of type
	 * {@link UncertainAdditiveUtilitySpace} and then generates a UncertainModel
	 * 
	 * @throws RepositoryException
	 * @throws NegotiatorException
	 * @throws InstantiateException
	 */
	@Test
	public void testUncertainAdditive() throws RepositoryException,
			NegotiatorException, InstantiateException {
		Domain domain = mock(Domain.class);

		UncertainAdditiveUtilitySpace utilspace = mock(
				UncertainAdditiveUtilitySpace.class);
		when(utilspace.copy()).thenReturn(utilspace);
		when(utilspace.getDomain()).thenReturn(domain);

		PartyRepItem partyrep = mock(PartyRepItem.class);
		ProfileRepItem profilerep = mock(ProfileRepItem.class);
		when(profilerep.create()).thenReturn(utilspace);

		Session session = mock(Session.class);
		when(session.getDeadlines()).thenReturn(mock(Deadline.class));
		when(session.getTimeline()).thenReturn(mock(Timeline.class));

		SessionsInfo info = mock(SessionsInfo.class);
		when(info.getPersistentDataType())
				.thenReturn(PersistentDataType.DISABLED);

		AgentID agentid = mock(AgentID.class);

		NegotiationParty party = mock(NegotiationParty.class);
		when(partyrep.load()).thenReturn(party);

		NegotiationPartyInternal partyint = new NegotiationPartyInternal(
				partyrep, profilerep, session, info, agentid);

		assertNotNull(partyint.getUncertainModel());

	}
}
