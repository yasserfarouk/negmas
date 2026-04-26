package genius.core.actions;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Vote;
import genius.core.parties.Mediator;
import genius.core.protocol.SimpleMediatorBasedProtocol;

/**
 * informs about a voting result. This action is created by {@link Mediator}s to
 * inform the others about the current state. Also used by the
 * {@link SimpleMediatorBasedProtocol} to find the last agreement. Immutable.
 * 
 * @author Reyhan
 */

@XmlRootElement
public class InformVotingResult extends DefaultActionWithBid {
	@XmlElement
	protected Vote vote;

	public InformVotingResult(AgentID party, Bid bid, Vote vote) {
		super(party, bid);
		if (vote == null) {
			throw new NullPointerException("vote=null");
		}
		this.vote = vote;
	}

	/**
	 * 
	 * @return a non-null vote
	 */
	public Vote getVotingResult() {
		return vote;
	}

	public String toString() {
		return "Voting Result: " + (vote == null ? "null" : (vote == Vote.ACCEPT ? "Accept" : "Reject"));
	}

}
