package negotiator.protocol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.protocol.AlternatingMajorityConsensusProtocol;
import genius.core.protocol.AlternatingMultipleOffersProtocol;
import genius.core.session.Round;

public class AlternatingMajorityConsensusProtocolTest extends AlternatingMultipleOffersProtocolTest {

	@Override
	protected Class<? extends AlternatingMultipleOffersProtocol> getProtocol() {
		return AlternatingMajorityConsensusProtocol.class;
	}

	@Override
	@Test
	public void testIsNotFinishedWithIncompleteVotes() {

		// We need at least two rounds for a finish.
		List<Round> rounds = new ArrayList<Round>();

		// 3 turns is apparently insufficient. Not clear why.
		// but protocol.isFinished should not crash on that.
		Offer firstoffer = offer();
		Offer secondoffer = offer();
		rounds.add(mockedRound(new Action[] { firstoffer, secondoffer, offer() }));
		// incomplete: only votes for bid 1.
		Round voteRound = mockedRound(new Action[] { REJECT, ACCEPT, ACCEPT });
		rounds.add(voteRound);

		when(session.getMostRecentRound()).thenReturn(voteRound);
		when(session.getRounds()).thenReturn(rounds);

		// we have round 0 and 1 done
		when(session.getRoundNumber()).thenReturn(2);
		when(session.getRounds()).thenReturn(rounds);

		assertFalse(protocol.isFinished(session, parties));
		assertEquals(secondoffer.getBid(), protocol.getCurrentAgreement(session, parties));
	}

}
